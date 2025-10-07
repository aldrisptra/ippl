from telegram import Bot

# Pakai token bot yang tadi berhasil di /getMe
TELEGRAM_TOKEN = "7697921487:AAEvZXLkC61Nzx-eh1e2BES1VfqSJ3wN32E"
CHAT_ID = "1215968232"  # chat_id kamu

bot = Bot(token=TELEGRAM_TOKEN)

bot.send_message(chat_id=CHAT_ID, text="✅ Tes dari projek aldri nih ces!")
