#!/usr/bin/env python
"""
Generic entry point for running any module in src/ with accelerate launch.

Usage:
    accelerate launch run.py src.sr.1_train_model --your-args
    accelerate launch run.py src.dblocking.1_train_model --your-args
    accelerate launch run.py src.sr.2_compress_lut_from_net --your-args

Multi-GPU:
    accelerate launch --multi_gpu --num_processes=4 run.py src.sr.1_train_model --args

Telegram notifications:
    Set environment variables TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to receive
    notifications when training finishes or fails.
"""

import asyncio
import html
import os
import runpy
import sys
import traceback

from dotenv import load_dotenv

load_dotenv()

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


def send_telegram_message(message: str) -> bool:
    """Send a message via Telegram bot.

    Requires TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables.
    Returns True if message was sent successfully, False otherwise.
    """
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not bot_token or not chat_id:
        return False

    try:
        import telegram
    except ImportError:
        print(
            "python-telegram-bot not installed. Run: pip install python-telegram-bot",
            file=sys.stderr,
        )
        return False

    # Truncate message if too long (Telegram limit is 4096 characters)
    if len(message) > 4000:
        message = message[:2000] + "\n\n... [truncated] ...\n\n" + message[-1900:]

    async def _send():
        bot = telegram.Bot(token=bot_token)
        await bot.send_message(chat_id=chat_id, text=message, parse_mode="HTML")

    try:
        asyncio.run(_send())
        return True
    except Exception as e:
        print(f"Failed to send Telegram message: {e}", file=sys.stderr)
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: accelerate launch run.py <module_path> [args...]")
        print("\nExamples:")
        print("  accelerate launch run.py src.sr.1_train_model --scale 4 --modes ss")
        print(
            "  accelerate launch run.py src.sr.2_compress_lut_from_net --model SPF_LUT_net"
        )
        print("  accelerate launch run.py src.dblocking.1_train_model --qf 10")
        print("\nMulti-GPU:")
        print(
            "  accelerate launch --multi_gpu --num_processes=4 run.py src.sr.1_train_model"
        )
        print("\nTelegram notifications:")
        print("  Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables")
        sys.exit(1)

    # Get the module to run
    module_name = sys.argv[1]
    original_args = " ".join(sys.argv[1:])

    # Remove the module name from sys.argv so the target script sees correct args
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    # Run the module
    try:
        runpy.run_module(module_name, run_name="__main__")
        # Success notification
        send_telegram_message(
            f"<b>Training completed successfully</b>\n\n"
            f"Args: <code>{original_args}</code>"
        )
    except ModuleNotFoundError as e:
        print(f"Error: Could not find module '{module_name}'")
        print(f"Details: {e}")
        print("\nMake sure the module path is correct (e.g., src.sr.1_train_model)")
        sys.exit(1)
    except KeyboardInterrupt:
        send_telegram_message(
            f"<b>Training interrupted (KeyboardInterrupt)</b>\n\n"
            f"<code>{module_name}</code>\n"
            f"Args: <code>{original_args}</code>"
        )
        sys.exit(130)
    except Exception as e:
        # Get full traceback
        tb = traceback.format_exc()
        error_msg = (
            f"<b>Training failed</b>\n\n"
            f"Args: <code>{original_args}</code>\n\n"
            f"<b>Error:</b> {type(e).__name__}: {e}\n\n"
            f"<b>Traceback:</b>\n<pre>{html.escape(tb)}</pre>"
        )
        send_telegram_message(error_msg)
        # Re-raise to preserve original behavior
        raise
