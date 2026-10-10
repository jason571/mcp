// Trusted compiler adapter. Executed only inside the isolated container.
// Uploaded code, npm scripts, compiler plugins and custom loaders are never run here.
'use strict';
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const ts = require('/opt/exoharness-typescript/node_modules/typescript');
const tooling = '/opt/exoharness-typescript/node_modules';
const tsxCli = path.join(tooling, 'tsx/dist/cli.mjs');
const semver = require('/usr/local/lib/node_modules/npm/node_modules/semver');
const suffix = /\.(?:ts|tsx|mts|cts)$/;
const excluded = new Set(['node_modules', '.git', '__pycache__', '.cache']);
const json = p => JSON.parse(fs.readFileSync(p, 'utf8'));
const digest = s => crypto.createHash('sha256').update(s).digest('hex');
const versions = {node: process.version, typescript: ts.version, tsx: json(path.join(tooling, 'tsx/package.json')).version,
  adapter: digest(fs.readFileSync(__filename))};

function scan(root) {
  const result = [];
  function visit(dir) {
    for (const entry of fs.readdirSync(dir, {withFileTypes: true}).sort((a,b) => a.name.localeCompare(b.name))) {
      if (excluded.has(entry.name)) continue;
      const file = path.join(dir, entry.name);
      if (entry.isSymbolicLink()) throw Error('TypeScript source must not contain symlinks');
      if (entry.isDirectory()) visit(file);
      else if (entry.isFile()) result.push(path.relative(root, file));
      else throw Error('TypeScript source must contain regular files');
      if (result.length > 4096) throw Error('TypeScript package exceeds 4096 files');
    }
  }
  visit(root);
  return result;
}

function sourceHash(root) {
  const hash = crypto.createHash('sha256');
  for (const file of scan(root)) hash.update(file + '\0').update(fs.readFileSync(path.join(root, file)));
  return hash.digest('hex');
}

function links(destination, roots) {
  fs.mkdirSync(destination, {recursive:true});
  for (const root of roots) {
    if (!fs.existsSync(root)) continue;
    for (const name of fs.readdirSync(root)) {
      if (name.startsWith('.')) continue;
      if (name.startsWith('@')) {
        links(path.join(destination, name), [path.join(root, name)]);
      } else if (!fs.existsSync(path.join(destination, name))) {
        fs.symlinkSync(path.join(root, name), path.join(destination, name));
      }
    }
  }
}

function configuration(root) {
  const manifestFile = path.join(root, 'package.json');
  const manifest = fs.existsSync(manifestFile) ? json(manifestFile) : {};
  if (manifest.engines?.node && !semver.satisfies(process.version, manifest.engines.node))
    throw Error('Node version does not satisfy package.json engines.node: ' + manifest.engines.node);
  const dependencies = {...manifest.devDependencies, ...manifest.dependencies};
  for (const name of ['typescript', 'tsx']) {
    if (dependencies[name] && !semver.satisfies(versions[name], dependencies[name]))
      throw Error(`Managed ${name} ${versions[name]} does not satisfy declared ${dependencies[name]}`);
  }
  let mode = manifest.exoharness?.typescript?.mode;
  if (mode === undefined) {
    const build = manifest.scripts?.build;
    if (build && !/^tsc(?:\s+(?:-p|--project)\s+\.?\/?tsconfig\.json)?\s*$/.test(build))
      throw Error('Custom build script is not executed during import. Use a standard tsc build or explicitly select exoharness.typescript.mode=tsx.');
    mode = build ? 'tsc' : 'tsx';
  }
  if (!['tsx', 'tsc'].includes(mode)) throw Error('exoharness.typescript.mode must be tsx or tsc');
  const configFile = path.join(root, 'tsconfig.json');
  let parsed;
  if (fs.existsSync(configFile)) {
    const config = ts.readConfigFile(configFile, ts.sys.readFile);
    if (config.error) fail([config.error]);
    parsed = ts.parseJsonConfigFileContent(config.config, ts.sys, root, undefined, configFile);
  } else {
    parsed = {options: {target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.NodeNext,
      moduleResolution: ts.ModuleResolutionKind.NodeNext, jsx: ts.JsxEmit.ReactJSX,
      esModuleInterop: true, skipLibCheck: true}, fileNames: [], errors: []};
  }
  if (parsed.errors.length) fail(parsed.errors);
  if (parsed.projectReferences?.length) throw Error('Multi-project TypeScript references require a separately prepared build; this runtime checks one project.');
  if (parsed.options.plugins?.length) throw Error('Custom TypeScript compiler plugins are not enabled in the managed runtime');
  if (parsed.options.emitDecoratorMetadata && mode !== 'tsc') throw Error('emitDecoratorMetadata requires exoharness.typescript.mode=tsc');
  // Check every shipped TS source, even if a tsconfig exclude would hide an entrypoint.
  const files = [...new Set([...parsed.fileNames, ...scan(root).filter(p => suffix.test(p)).map(p => path.join(root,p))])];
  if (!files.length) throw Error('No TypeScript source files were found');
  if (files.length > 512) throw Error('TypeScript compilation exceeds 512 source files');
  return {mode, configFile: fs.existsSync(configFile) ? configFile : null, files,
    options: {...parsed.options, noEmit: true, incremental: false, composite: false,
      typeRoots: parsed.options.typeRoots ?? [path.join(root, 'node_modules/@types'), path.resolve(root, '../node_modules/@types')]}};
}

function fail(diagnostics) {
  throw Error(ts.formatDiagnostics(diagnostics.slice(0, 12), {
    getCanonicalFileName: f => f, getCurrentDirectory: () => process.cwd(), getNewLine: () => '\n',
  }).slice(0, 6000));
}

function check(root, cache, emit) {
  const config = configuration(root);
  const outputs = {};
  const options = {...config.options};
  if (config.mode === 'tsc') Object.assign(options, {noEmit: !emit, noEmitOnError:true,
    emitDeclarationOnly:false, declaration:false, declarationMap:false, sourceMap:true,
    outFile:undefined, rootDir:root, outDir:path.join(cache, 'build'), rewriteRelativeImportExtensions:true});
  const program = ts.createProgram(config.files, options);
  const diagnostics = ts.getPreEmitDiagnostics(program);
  if (diagnostics.length) fail(diagnostics);
  if (config.mode === 'tsc' && emit) {
    const result = program.emit(undefined, (file, data, bom, onError, sources) => {
      const target = path.resolve(file);
      if (!target.startsWith(path.resolve(cache, 'build') + path.sep)) throw Error('TypeScript output escaped managed build directory');
      fs.mkdirSync(path.dirname(target), {recursive:true});
      fs.writeFileSync(target, data);
      if (/\.(?:js|mjs|cjs)$/.test(target)) for (const source of sources ?? [])
        outputs[path.relative(root, source.fileName)] = target;
    });
    if (result.emitSkipped || result.diagnostics.length) fail(result.diagnostics);
    // Preserve the package's module system and runtime data without copying source over emitted JS.
    for (const file of scan(root).filter(p => !suffix.test(p))) {
      const target = path.join(cache, 'build', file);
      if (!fs.existsSync(target)) {
        fs.mkdirSync(path.dirname(target), {recursive:true});
        fs.copyFileSync(path.join(root,file), target);
      }
    }
  }
  return {mode:config.mode, config:config.configFile, outputs, files:config.files.length, versions};
}

function prepare(source, target) {
  const cache = path.join(target, 'typescript');
  const hash = sourceHash(source);
  fs.mkdirSync(cache, {recursive:true});
  const root = path.join(cache, 'project');
  if (!fs.existsSync(root)) fs.cpSync(source, root, {recursive:true, filter: p => !path.relative(source,p).split(path.sep).some(v => excluded.has(v))});
  if (sourceHash(root) !== hash) throw Error('TypeScript preparation source snapshot changed');
  links(path.join(cache,'node_modules'), [path.join(target,'node/node_modules'), '/usr/local/lib/node_modules', tooling]);
  const result = {...check(root, cache, true), source_sha256: hash, status:'ready'};
  fs.writeFileSync(path.join(cache,'prepared.json'), JSON.stringify(result));
  return result;
}

function verify(source, target) {
  const cache = path.join(target,'typescript');
  const result = json(path.join(cache,'prepared.json'));
  if (result.status !== 'ready' || JSON.stringify(result.versions) !== JSON.stringify(versions)
      || result.source_sha256 !== sourceHash(source) || result.source_sha256 !== sourceHash(path.join(cache,'project')))
    throw Error('TypeScript preparation snapshot or toolchain changed');
  check(path.join(cache,'project'), cache, false);
  for (const file of Object.values(result.outputs)) if (!fs.existsSync(file)) throw Error('Prepared TypeScript output is missing');
  return result;
}

function plan(source, target, entry) {
  if (!suffix.test(entry) || path.isAbsolute(entry) || entry.split(/[\\/]/).includes('..')) throw Error('Invalid TypeScript entrypoint');
  const cached = source === '/skill';
  let result, root, cache;
  if (cached) {
    cache = path.join(target,'typescript');
    root = path.join(cache,'project');
    result = json(path.join(cache,'prepared.json'));
    if (result.status !== 'ready' || JSON.stringify(result.versions) !== JSON.stringify(versions)
        || result.source_sha256 !== sourceHash(source)) throw Error('Skill TypeScript runtime must be prepared for this source and toolchain');
  } else {
    root = source;
    cache = '/tmp/exoharness-typescript';
    links(path.join(cache,'node_modules'), [path.join(target,'node/node_modules'), '/usr/local/lib/node_modules', tooling]);
    // Node ESM resolution needs a real ancestor node_modules; NODE_PATH is not sufficient.
    const modules = path.join(root,'node_modules');
    if (fs.existsSync(modules)) throw Error('Task node_modules is managed by the sandbox, not task input');
    fs.symlinkSync(path.join(cache,'node_modules'), modules);
    result = check(root, cache, true);
  }
  const file = result.mode === 'tsc' ? result.outputs[entry] : path.join(root, entry);
  if (!file || !fs.existsSync(file)) throw Error('TypeScript entrypoint has no runnable output');
  const command = [process.execPath, tsxCli];
  if (result.config) command.push('--tsconfig', result.config);
  else command.push('--tsconfig', '/opt/exoharness-typescript/tsconfig.json');
  return [...command, file];
}

try {
  const [mode, source, target, entry] = process.argv.slice(2);
  let result;
  if (mode === 'prepare') result = prepare(source, target);
  else if (mode === 'check') result = verify(source, target);
  else if (mode === 'plan') result = plan(source, target, entry);
  else throw Error('Invalid managed TypeScript operation');
  process.stdout.write(JSON.stringify(result));
} catch (error) {
  process.stderr.write(String(error.message).slice(0, 6000) + '\n');
  process.exitCode = 1;
}
