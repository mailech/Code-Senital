import asyncio
from app.services.github_client import create_file_and_pr


async def main() -> None:
    pr = await create_file_and_pr(
        branch="sentinel/add-sample-file",
        path="docs/SENTINEL_DEMO.md",
        content_text="# Sentinel Demo\n\nThis file was added by the Self-Healing Codebase Sentinel demo.",
        title="Sentinel: add sample file",
        body="Adding a sample file to verify PR flow.",
    )
    print(pr)


if __name__ == "__main__":
    asyncio.run(main())


