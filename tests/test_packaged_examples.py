"""在无私人资料的新目录运行教学入口，检查公开示例和本地覆盖。"""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class PackagedExamplesTests(unittest.TestCase):
    def run_entry(self, local_chunk=None):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            shutil.copyfile(ROOT / 'main.py', root / 'main.py')
            (root / 'examples').mkdir()
            shutil.copyfile(ROOT / 'examples' / 'knowledge_chunks.jsonl',
                            root / 'examples' / 'knowledge_chunks.jsonl')
            if local_chunk is not None:
                (root / 'data').mkdir()
                (root / 'data' / 'knowledge_chunks.jsonl').write_text(
                    json.dumps(local_chunk, ensure_ascii=False) + '\n', encoding='utf-8')
            return subprocess.run([sys.executable, '-X', 'utf8', '-B', str(root / 'main.py')],
                                  cwd=root, capture_output=True, text=True, encoding='utf-8', timeout=15)

    def test_clean_checkout_runs_with_packaged_examples(self):
        result = self.run_entry()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('分块数量：100', result.stdout)
        self.assertIn('KB-001', result.stdout)

    def test_existing_local_chunks_take_precedence(self):
        chunk = {'text': '本地合成资料', 'metadata': {'knowledge_id': 'LOCAL-ONLY', 'source': 'fixture'}}
        result = self.run_entry(chunk)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('分块数量：1', result.stdout)
        self.assertIn('LOCAL-ONLY', result.stdout)
        self.assertNotIn('KB-001', result.stdout)


if __name__ == '__main__':
    unittest.main()
