import unittest
from lesson_renderer.artifact_input import addresses
from lesson_renderer.schema import AudioBinding

class ArtifactInputTests(unittest.TestCase):
    def test_returned_signed_url_and_legacy_attachment_are_distinct(self):
        url="http://127.0.0.1:9790/artifacts/artifact-"+"a"*64+"/intro.mp3?expires=0&signature="+"b"*64
        source,meta,identity=addresses(url,"http://127.0.0.1:9790")
        self.assertEqual(source,url)
        self.assertIn("intro.mp3/metadata?",meta)
        self.assertEqual(identity,"artifact-"+"a"*64)
        AudioBinding(segment_id="intro",artifact_url=url)
        AudioBinding(segment_id="intro",attachment_id="att_"+"a"*64)
        for value in [{"segment_id":"intro"},{"segment_id":"intro","attachment_id":"att_"+"a"*64,"artifact_url":url}, {"segment_id":"intro","attachment_id":"artifact-"+"a"*64}]:
            with self.assertRaises(ValueError): AudioBinding.model_validate(value)

    def test_untrusted_origin_redirect_paths_unsigned_and_modified_references_rejected(self):
        origin="http://127.0.0.1:9790"
        valid=origin+"/artifacts/artifact-"+"a"*64+"/intro.mp3?expires=0&signature="+"b"*64
        invalid=[valid.replace("9790","9797"),valid.replace("http:","https:"),valid.replace("127.0.0.1","example.com"),valid.replace("/artifacts/","/v1/attachment-assets/"),valid.replace("intro.mp3","%2Fetc%2Fpasswd"),valid+"#fragment",valid+"&signature="+"c"*64,valid+"&download=true",valid.replace("signature=","signature=x"),valid.split("?")[0]]
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaises(ValueError): addresses(value,origin)
