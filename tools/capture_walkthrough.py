"""Capture the running PUBLIC demo UI with a clean headless browser.

Optional local tool: pip install playwright; use an installed Edge browser.
It never injects answers, mocks providers, edits the DOM or runs a judge.
"""

import argparse
import json
from pathlib import Path
import time
from urllib.parse import urlparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8512')
    parser.add_argument('--output', type=Path, default=Path('docs/assets/walkthrough'))
    parser.add_argument('--submit', action='store_true', help='Run public queries using the configured application provider')
    args = parser.parse_args()
    if urlparse(args.url).hostname not in {'127.0.0.1', 'localhost'}:
        raise ValueError('Only the local public demo may be captured')
    from playwright.sync_api import sync_playwright

    args.output.mkdir(parents=True, exist_ok=True)
    evidence = {'kind': 'actual_streamlit_ui', 'mocked': False, 'judge_calls': 0, 'steps': []}
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(channel='msedge', headless=True)
        page = browser.new_page(viewport={'width': 1440, 'height': 1100}, device_scale_factor=1)
        public_verified = False
        try:
            page.goto(args.url)
            page.get_by_text('Public walkthrough', exact=False).wait_for(timeout=90000)
            public_verified = True
            chat = page.get_by_placeholder('Ask about past projects, tech stacks, proposals...')
            if not chat.is_enabled():
                raise RuntimeError('Real public application is not ready; refuse staged capture')
            page.screenshot(path=str(args.output / '01-public-ready.png'))
            evidence['steps'].append({'step': 'startup', 'screenshot': '01-public-ready.png', 'status': 'ready'})
            if args.submit:
                query = 'Which projects used Microsoft Azure?'
                chat.fill(query)
                chat.press('Enter')
                page.get_by_text('Why this retrieval path?', exact=True).wait_for(timeout=120000, state='attached')
                expanders = page.locator('[data-testid="stExpander"]')
                expanders.last.locator('summary').click()
                explanation = page.get_by_text('Why this retrieval path?', exact=True)
                explanation.wait_for(timeout=30000)
                page.screenshot(path=str(args.output / '02-azure-answer.png'))
                explanation.scroll_into_view_if_needed()
                page.screenshot(path=str(args.output / '03-retrieval-explanation.png'))
                # The trace text is public output, not a claim-level provenance proof.
                public_text = page.locator('[data-testid="stChatMessage"]').last.inner_text()
                evidence['steps'].append({'step': 'azure_relationship', 'question': query, 'public_ui_text': public_text})
                grounding = page.get_by_text('final_grounding_verifier', exact=False).last
                if grounding.count():
                    grounding.scroll_into_view_if_needed()
                    page.screenshot(path=str(args.output / '04-grounding-check.png'))
                else:
                    evidence['steps'][-1]['grounding_screenshot'] = 'not_available'
                # Stop rather than manufacture a successful walkthrough on provider failure.
                if 'Rate limit reached' in public_text or 'Request failed' in public_text:
                    evidence['steps'][-1]['status'] = 'provider_or_pipeline_error'
            evidence['status'] = 'captured'
        except Exception as exc:
            evidence.update({'status': 'incomplete', 'error_type': type(exc).__name__})
            if public_verified:
                page.screenshot(path=str(args.output / 'incomplete-public-run.png'))
            raise
        finally:
            evidence['captured_at_unix'] = time.time()
            (args.output / 'capture.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
            browser.close()


if __name__ == '__main__':
    main()
