import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from skill_workspace.common import read, write, validate_request
from skill_workspace.runtime_installer import requirements, typescript_files
from skill_workspace.prepare_runtime import identity, process_pending
from skill_workspace.broker import docker, run_job, runtime_inventory
from skill_workspace.workspaces import Workspaces


class TypeScriptContracts(unittest.TestCase):
    def test_all_typescript_extensions_are_explicit_not_host_commands(self):
        for suffix in ('.ts', '.tsx', '.mts', '.cts'):
            validate_request({'workspace_id':'ws_'+'a'*64, 'workspace_revision':0, 'runtime':'typescript',
                'entrypoint':'main'+suffix, 'files':[], 'outputs':[]})
        with self.assertRaisesRegex(ValueError,'entrypoint'):
            validate_request({'workspace_id':'ws_'+'a'*64, 'workspace_revision':0, 'runtime':'typescript',
                'entrypoint':'main.sh', 'files':[], 'outputs':[]})
        with self.assertRaisesRegex(ValueError,'runtime=typescript'):
            validate_request({'workspace_id':'ws_'+'a'*64, 'workspace_revision':0, 'runtime':'node',
                'entrypoint':'main.ts', 'files':[], 'outputs':[]})

    def test_development_dependencies_for_ts_without_running_build_hooks(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'main.ts').write_text('throw new Error("must not run")')
            (root/'package.json').write_text(json.dumps({'dependencies':{'is-number':'7.0.0'},
                'devDependencies':{'@types/node':'24.3.0'},'scripts':{'postinstall':'exit 1','build':'tsc'}}))
            self.assertEqual(len(typescript_files(root)),1)
            self.assertEqual(requirements(root)[1],{'is-number':'7.0.0','@types/node':'24.3.0'})
            (root/'main.ts').unlink()
            self.assertEqual(requirements(root)[1],{'is-number':'7.0.0'})


@unittest.skipUnless(os.getenv('SKILL_WORKSPACE_INTEGRATION')=='1','requires built Docker runtime')
class TypeScriptDocker(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(prefix='exoharness-typescript-test-')
        self.root=Path(self.temporary.name)
        self.image=os.environ['WORKSPACE_IMAGE_ID']
        self.env=patch.dict(os.environ,{'WORKSPACE_IMAGE_ID':self.image})
        self.env.start()
        write(self.root/'worker.json',{'image':self.image})
        write(self.root/'task-code.json',{'enabled':True,'image':self.image})
        self.jobs=[]

    def tearDown(self):
        for job in self.jobs: docker('rm','-f','exoharness-'+job.name,check=False)
        self.env.stop()
        self.temporary.cleanup()

    def job(self,files,entry='main.ts',outputs=None,workspace=None,package=None):
        store=Workspaces(self.root)
        if workspace is None:
            workspace=store.create(*(package or (None,None)),runtime_image=self.image)['workspace_id']
        state=store.state(workspace)
        job=self.root/'jobs'/('work_'+os.urandom(32).hex())
        data={'workspace_id':workspace,'workspace_revision':state['revision'],'runtime_image':self.image,
            'package_name':state['package_name'],'package_digest':state['package_digest'],
            'runtime':'typescript','entrypoint':entry,'entrypoint_root':'skill' if package else 'workspace',
            'files':[{'path':k,'content':v} for k,v in files.items()], 'outputs':outputs or [], 'arguments':['literal argument']}
        write(job/'request.json',data)
        write(job/'input/request.json',data)
        (job/'input/request.json').chmod(0o644)
        previous=[]
        for index,(name,item) in enumerate(state['files'].items()):
            blob=store.bytes(workspace,name,state)
            source='previous-'+str(index)
            (job/'input'/source).write_bytes(blob)
            previous.append({'file':source,'path':name,'size_bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest()})
        write(job/'input/files.json',previous)
        (job/'input/files.json').chmod(0o644)
        write(job/'state.json',{'status':'queued'})
        state['active_job']=job.name
        write(store.directory(workspace)/'state.json',state)
        self.jobs.append(job)
        return job,workspace

    def execute(self,job):
        run_job(job,self.root/'packages',self.image,self.root/'worker.json')

    def prepare(self,files):
        name='unrelated-ts-fixture'
        digest=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
        package=self.root/'packages'/name/digest
        package.mkdir(parents=True)
        (package/'SKILL.md').write_text('---\nname: unrelated-ts-fixture\ndescription: Test fixture\n---\nOriginal instructions.\n')
        for file,content in files.items():
            (package/file).parent.mkdir(parents=True,exist_ok=True)
            (package/file).write_text(content)
        key=identity(name,digest,self.image)
        queued=self.root/'imports'/key
        write(queued/'request.json',{'package_name':name,'package_digest':digest,'image':self.image})
        write(queued/'state.json',{'status':'queued'})
        process_pending(self.root,self.root/'packages',self.image,self.root/'worker.json')
        return (name,digest),queued,package

    def test_real_typescript_and_tsx_with_modules_and_artifacts(self):
        inventory=runtime_inventory(self.image)
        self.assertTrue(all(inventory['commands'][name] for name in ['node','tsc','tsx']))
        for ext in ['ts','mts','cts']:
            with self.subTest(extension=ext):
                job,_=self.job({'main.'+ext:"import {writeFileSync} from 'node:fs'; enum Answer {Value=42}; writeFileSync('output.txt', String(Answer.Value)+':'+process.argv[2]);"},entry='main.'+ext,outputs=['output.txt'])
                self.execute(job)
                self.assertEqual((job/'published/output.txt').read_text(),'42:literal argument')
        job,_=self.job({'main.tsx':"import React from 'react'; import {renderToStaticMarkup} from 'react-dom/server'; import {writeFileSync} from 'node:fs'; writeFileSync('output.txt',renderToStaticMarkup(<h1>TypeScript JSX</h1>));"},entry='main.tsx',outputs=['output.txt'])
        self.execute(job)
        self.assertEqual((job/'published/output.txt').read_text(),'<h1>TypeScript JSX</h1>')

    def test_type_error_never_runs_code_and_can_be_patched(self):
        job,workspace=self.job({'main.ts':"import {writeFileSync} from 'node:fs'; const answer: number = 'bad'; writeFileSync('output.txt',String(answer));"})
        with self.assertRaisesRegex(ValueError,'not assignable'):
            self.execute(job)
        self.assertNotIn('output.txt',Workspaces(self.root).state(workspace)['files'])
        saved=Workspaces(self.root).inspect(workspace,'main.ts')
        write(job/'state.json',{'status':'failed'})
        Workspaces(self.root).edit(workspace,saved['revision'],[],[{'path':'main.ts','sha256':saved['sha256'],'before':"'bad'",'after':'42'}],[])
        repaired,_=self.job({},workspace=workspace,outputs=['output.txt'])
        self.execute(repaired)
        self.assertEqual((repaired/'published/output.txt').read_text(),'42')

    def test_real_skill_dependency_alias_esm_cache_and_no_import_side_effect(self):
        files={'package.json':json.dumps({'type':'module','dependencies':{'is-number':'7.0.0'},'devDependencies':{'@types/node':'24.3.0'},'scripts':{'postinstall':'touch /runtime/MUST_NOT_EXIST'}}),
            'tsconfig.json':json.dumps({'compilerOptions':{'module':'ESNext','moduleResolution':'Bundler','target':'ES2022','baseUrl':'.','paths':{'@lib/*':['lib/*']},'noEmit':True}}),
            'lib/answer.ts':'export const answer: number=42;',
            'types.d.ts':"declare module 'is-number' {export default function isNumber(value: unknown): boolean;}",
            'scripts/main.ts':"import {writeFileSync} from 'node:fs'; import isNumber from 'is-number'; import {answer} from '@lib/answer'; writeFileSync('output.txt',String(isNumber(answer)));"}
        package,queued,source=self.prepare(files)
        self.assertEqual(read(queued/'state.json')['status'],'ready',read(queued/'state.json'))
        self.assertFalse((queued/'runtime/MUST_NOT_EXIST').exists())
        self.assertFalse((source/'output.txt').exists())
        self.assertFalse(read(queued/'runtime/verified.json')['business_code_executed'])
        stamp=(queued/'runtime/typescript/prepared.json').stat().st_mtime_ns
        with patch('skill_workspace.prepare_runtime.subprocess.Popen',side_effect=AssertionError('must reuse prepared snapshot')):
            process_pending(self.root,self.root/'packages',self.image,self.root/'worker.json')
        fake_node=queued/'runtime/node/node_modules/.bin/node'
        fake_node.parent.mkdir(parents=True,exist_ok=True)
        fake_node.write_text('#!/bin/sh\nprintf bypassed > /workspace/bypassed\nexit 88\n')
        fake_node.chmod(0o755)
        job,workspace=self.job({},entry='scripts/main.ts',package=package,outputs=['output.txt'])
        self.execute(job)
        self.assertEqual((job/'published/output.txt').read_text(),'true')
        self.assertNotIn('bypassed',Workspaces(self.root).state(workspace)['files'])
        self.assertEqual((queued/'runtime/typescript/prepared.json').stat().st_mtime_ns,stamp)
        self.assertEqual((source/'scripts/main.ts').read_text(),files['scripts/main.ts'])

    def test_real_standard_tsc_build_is_cached_not_custom_script_execution(self):
        package,queued,_=self.prepare({'package.json':'{"scripts":{"build":"tsc -p tsconfig.json"}}',
            'tsconfig.json':'{"compilerOptions":{"target":"ES2022","module":"CommonJS","experimentalDecorators":true,"emitDecoratorMetadata":true}}',
            'main.ts':"import {writeFileSync} from 'node:fs'; function mark(...args: any[]) {} class Item { @mark value: string='built'; } writeFileSync('output.txt',new Item().value);"})
        self.assertEqual(read(queued/'state.json')['status'],'ready',read(queued/'state.json'))
        metadata=read(queued/'runtime/typescript/prepared.json')
        self.assertEqual(metadata['mode'],'tsc')
        self.assertIn('__metadata',(queued/'runtime/typescript/build/main.js').read_text())
        job,_=self.job({},package=package,outputs=['output.txt'])
        self.execute(job)
        self.assertEqual((job/'published/output.txt').read_text(),'built')

    def test_invalid_skill_is_blocked_without_execution(self):
        _,queued,source=self.prepare({'main.ts':"const a: number = 'bad'; throw Error('business code');"})
        self.assertEqual(read(queued/'state.json')['status'],'failed')
        self.assertIn('not assignable',read(queued/'state.json')['message'])
        self.assertFalse((queued/'runtime/verified.json').exists())

    def test_custom_build_and_incompatible_toolchain_are_not_silently_accepted(self):
        for manifest,reason in [({'scripts':{'build':'node evil.js'}},'Custom build script'),
                                ({'engines':{'node':'>=999'}},'engines.node')]:
            with self.subTest(manifest=manifest):
                _,queued,_=self.prepare({'package.json':json.dumps(manifest),'main.ts':'console.log(42)',
                    'evil.js':"require('node:fs').writeFileSync('/runtime/EVIL','bad')"})
                self.assertEqual(read(queued/'state.json')['status'],'failed')
                self.assertIn(reason,read(queued/'state.json')['message'])
                self.assertFalse((queued/'runtime/EVIL').exists())

    def test_lockfile_and_cached_toolchain_identity(self):
        files={'package.json':'{"dependencies":{"is-number":"7.0.0"}}','main.ts':'console.log(42)'}
        _,first,_=self.prepare(files)
        self.assertEqual(read(first/'state.json')['status'],'ready',read(first/'state.json'))
        files['package-lock.json']=(first/'runtime/node/package-lock.json').read_text()
        package,second,_=self.prepare(files)
        self.assertEqual(read(second/'state.json')['status'],'ready',read(second/'state.json'))
        self.assertEqual(read(second/'runtime/node/package-lock.json')['packages']['node_modules/is-number']['version'],'7.0.0')
        metadata=read(second/'runtime/typescript/prepared.json')
        metadata['versions']['typescript']='0.0.0'
        write(second/'runtime/typescript/prepared.json',metadata)
        (second/'runtime/typescript/prepared.json').chmod(0o644)
        job,_=self.job({},package=package)
        with self.assertRaisesRegex(ValueError,'toolchain'):
            self.execute(job)
        lock=json.loads(files['package-lock.json'])
        lock['packages']['node_modules/is-number']['resolved']='https://untrusted.invalid/package.tgz'
        files['package-lock.json']=json.dumps(lock)
        _,third,_=self.prepare(files)
        self.assertEqual(read(third/'state.json')['status'],'failed')
        self.assertIn('standard npm registry',read(third/'state.json')['message'])

    def test_typescript_obeys_same_network_readonly_uid_and_cancel_policy(self):
        code="""import {existsSync,writeFileSync} from 'node:fs'; import {connect} from 'node:net';
if (process.getuid?.() !== 65532 || existsSync('/var/run/docker.sock')) throw Error('isolation failed');
try {writeFileSync('/etc/escaped','bad'); throw Error('root writable');} catch(e) {if ((e as NodeJS.ErrnoException).code !== 'EROFS') throw e;}
const socket=connect({host:'1.1.1.1',port:443}); socket.setTimeout(500);
socket.on('connect',()=>{throw Error('network open')}); socket.on('error',()=>socket.destroy()); socket.on('timeout',()=>socket.destroy());
setTimeout(()=>writeFileSync('output.txt','isolation verified'),1000);
"""
        probe,_=self.job({'main.ts':code},outputs=['output.txt'])
        self.execute(probe)
        self.assertEqual((probe/'published/output.txt').read_text(),'isolation verified')
        job,_=self.job({'main.ts':"setInterval(()=>console.log('running'),1000);"})
        errors=[]
        def execute():
            try:self.execute(job)
            except Exception as error:errors.append(error)
        thread=threading.Thread(target=execute)
        thread.start()
        try:
            for _ in range(100):
                if read(job/'state.json')['status']=='working':break
                time.sleep(.1)
            time.sleep(4)
            (job/'cancel').touch()
            thread.join(20)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors,[])
            self.assertEqual(read(job/'state.json')['status'],'cancelled')
            self.assertNotEqual(docker('inspect','exoharness-'+job.name,check=False).returncode,0)
        finally:
            (job/'cancel').touch()
            thread.join(20)


if __name__=='__main__':unittest.main()
