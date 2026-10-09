"""验证 fork 安装来源与 README 本地链接；不调用模型或网络。"""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_URL = 'https://github.com/aifuxi/oil-codex-title'


class ReadmeTests(unittest.TestCase):
    def test_install_instructions_use_current_fork_consistently(self):
        instructions = []
        for relative in ('README.md', 'docs/使用与边界.md'):
            with self.subTest(document=relative):
                document = (ROOT / relative).read_text(encoding='utf-8')
                blocks = re.findall(r'```text\n(.*?)\n```', document, re.DOTALL)
                self.assertEqual(len(blocks), 1)
                instruction = blocks[0]
                self.assertEqual(re.findall(r'https://github\.com/[^\s，。]+', instruction),
                                 [REPOSITORY_URL])
                self.assertNotIn('oil-oil/oil-codex-title', instruction)
                for requirement in ('完整', '不要只安装 Skill', '信任 Stop Hook', 'doctor', '实际触发'):
                    self.assertIn(requirement, instruction)
                instructions.append(instruction)
        self.assertEqual(instructions[0], instructions[1])

    def test_readme_local_links_exist(self):
        readme = (ROOT / 'README.md').read_text(encoding='utf-8')
        targets = re.findall(r'\]\(([^)]+)\)', readme)
        targets.extend(re.findall(r'<img\b[^>]*\bsrc="([^"]+)"', readme))
        for target in targets:
            if '://' in target or target.startswith('#'):
                continue
            with self.subTest(target=target):
                self.assertTrue((ROOT / target.split('#', 1)[0]).is_file())


if __name__ == '__main__':
    unittest.main()
