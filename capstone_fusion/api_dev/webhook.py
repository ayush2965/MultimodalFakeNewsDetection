# bot_api.py
import discord
from discord.ext import commands
from io import BytesIO
import requests
import os

# -----------------------------
# CONFIG
# -----------------------------
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")  # Replace with your bot token
API_BASE_URL = "http://127.0.0.1:8000"  # Replace with deployed API if needed

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# -----------------------------
# HELLO COMMAND
# -----------------------------
@bot.command()
async def hello(ctx):
    await ctx.send("Hello! I can detect fake news. Use `!predict` commands.")


# -----------------------------
# TEXT PREDICTION
# -----------------------------
@bot.command()
async def predict_text(ctx, *, text: str = None):
    """Predict if a text news is fake or real"""

    try:
        # ---- If no text provided, try reply message ----
        if text is None:
            ref = ctx.message.reference

            if ref and ref.message_id:
                replied_msg = await ctx.channel.fetch_message(ref.message_id)
                text = replied_msg.content

            if not text:
                await ctx.send("Reply to a text message or provide text.")
                return

        # ---- API request ----
        response = requests.post(
            f"{API_BASE_URL}/predict/text",
            data={"text": text}
        )

        data = response.json()

        if "result" not in data:
            await ctx.send(f"⚠️ Error: {data.get('error', 'Unknown error')}")
            return

        res = data["result"]
        sources = data.get("sources", [])

        verdict = res.get("final_outcome", "Unknown")
        subtitle = res.get("subtitle", "")
        confidence = res.get("text_confidence", 0)

        # ---- Color logic ----
        if "True" in verdict or "Real" in verdict:
            color = discord.Color.green()
        elif "Fake" in verdict:
            color = discord.Color.red()
        else:
            color = discord.Color.orange()

        embed = discord.Embed(
            title=f"Verdict: {verdict}",
            description=subtitle,
            color=color
        )

        # ---- Stats ----
        embed.add_field(
            name="Confidence",
            value=f"{confidence:.2f}%",
            inline=True
        )

        embed.add_field(
            name="Sources Found",
            value=res.get("sources_found", 0),
            inline=True
        )

        embed.add_field(
            name="Trusted Sources",
            value=res.get("trusted_count", 0),
            inline=True
        )

        embed.add_field(
            name="Debunked Claims",
            value=res.get("debunk_count", 0),
            inline=True
        )

        embed.add_field(
            name="SVE Verdict",
            value=res.get("sve_verdict", "Unknown"),
            inline=False
        )

        # ---- Source splitting ----
        if sources:
            MAX_FIELD_LEN = 1024
            chunk = ""
            field_index = 1

            for src in sources[:5]:
                title = src.get("title", "No title")
                link = src.get("link", "")
                entry = f"• [{title}]({link})\n"

                if len(chunk) + len(entry) > MAX_FIELD_LEN:
                    embed.add_field(
                        name="Top Sources" if field_index == 1 else "Top Sources (continued)",
                        value=chunk,
                        inline=False
                    )
                    chunk = entry
                    field_index += 1
                else:
                    chunk += entry

            if chunk:
                embed.add_field(
                    name="Top Sources" if field_index == 1 else "Top Sources (continued)",
                    value=chunk,
                    inline=False
                )

        embed.set_footer(text="Aletheia - Restrict Misinformation, Empower Truth")

        await ctx.send(embed=embed)

    except Exception as e:
        await ctx.send(f"⚠️ Exception occurred: {str(e)}")

# -----------------------------
# IMAGE PREDICTION (with embed)
# -----------------------------
# -----------------------------
# IMAGE PREDICTION (full embed)
# -----------------------------
@bot.command()
async def predict_image(ctx):
    """Predict if an image news is fake or real. Can be used on a replied message containing an image"""
    
    # 1️⃣ Check if user attached an image in the command itself
    attachment = None
    if ctx.message.attachments:
        attachment = ctx.message.attachments[0]
    
    # 2️⃣ If not, check if the command is a reply to another message
    elif ctx.message.reference:
        ref_msg = await ctx.channel.fetch_message(ctx.message.reference.message_id)
        if ref_msg.attachments:
            attachment = ref_msg.attachments[0]
    
    # 3️⃣ If no attachment found
    if not attachment:
        await ctx.send("Please attach an image or reply to a message that contains one.")
        return

    # 4️⃣ Send the image to the API
    try:
        img_bytes = await attachment.read()
        files = {"image": (attachment.filename, BytesIO(img_bytes), attachment.content_type)}
        response = requests.post(f"{API_BASE_URL}/predict/image", files=files)
        data = response.json()

        if "result" in data:
            res = data["result"]

            # Create embed
            embed = discord.Embed(
                title=f"Verdict: {res.get('final_outcome', 'Unknown')}",
                description=res.get("subtitle", ""),
                color=discord.Color.green() if "True" in res.get("final_outcome", "") else discord.Color.red()
            )

            embed.add_field(name="Confidence", value=f"{res.get('image_confidence', 0):.2f}%", inline=True)
            embed.add_field(name="Total Sources Found", value=res.get("sources_found", 0), inline=True)
            embed.add_field(name="Trusted Sources", value=res.get("trusted_count", 0), inline=True)
            embed.add_field(name="Debunked Claims", value=res.get("debunk_count", 0), inline=True)
            

            # Show image
            embed.set_image(url=attachment.url)
            embed.set_footer(text="Aletheia - Restrict Misinformation, Empower Truth")

            # Top sources (optional, up to 5)
            top_sources = res.get("top_sources", None)
            if top_sources and isinstance(top_sources, list):
                for idx, s in enumerate(top_sources[:5], start=1):
                    title = s.get("title", "No title")
                    link = s.get("link", "")
                    embed.add_field(name=f"Source {idx}", value=f"[{title}]({link})", inline=False)

            await ctx.send(embed=embed)
        else:
            await ctx.send(f"⚠️ Error: {data.get('error', 'Unknown error')}")
    except Exception as e:
        await ctx.send(f"⚠️ Exception occurred: {str(e)}")


# -----------------------------
# AUDIO PREDICTION
# -----------------------------
@bot.command()
async def predict_audio(ctx):
    """Predict news authenticity from audio attachment or replied audio"""

    attachment = None

    # --- Direct attachment ---
    if ctx.message.attachments:
        attachment = ctx.message.attachments[0]

    # --- Reply detection ---
    elif ctx.message.reference:
        ref_msg = await ctx.channel.fetch_message(ctx.message.reference.message_id)
        if ref_msg.attachments:
            attachment = ref_msg.attachments[0]

    if not attachment:
        await ctx.send("Attach or reply to an audio file.")
        return

    try:
        audio_bytes = await attachment.read()

        files = {
            "audio": (
                attachment.filename,
                BytesIO(audio_bytes),
                attachment.content_type
            )
        }

        response = requests.post(
            f"{API_BASE_URL}/predict/audio",
            files=files
        )

        data = response.json()

        if "result" not in data:
            await ctx.send(f"⚠️ Error: {data.get('error', 'Unknown error')}")
            return

        res = data["result"]
        sources = data.get("sources", [])

        verdict = res.get("final_outcome", "Unknown")
        subtitle = res.get("subtitle", "")
        confidence = res.get("text_confidence", 0)

        # --- Color logic ---
        if "True" in verdict or "Real" in verdict:
            color = discord.Color.green()
        elif "Fake" in verdict:
            color = discord.Color.red()
        else:
            color = discord.Color.orange()

        embed = discord.Embed(
            title=f"Verdict: {verdict}",
            description=subtitle,
            color=color
        )

        # --- Stats ---
        embed.add_field(
            name="Confidence",
            value=f"{confidence:.2f}%",
            inline=True
        )

        embed.add_field(
            name="Sources Found",
            value=res.get("sources_found", 0),
            inline=True
        )

        embed.add_field(
            name="Trusted Sources",
            value=res.get("trusted_count", 0),
            inline=True
        )

        embed.add_field(
            name="Debunked Claims",
            value=res.get("debunk_count", 0),
            inline=True
        )

        embed.add_field(
            name="SVE Verdict",
            value=res.get("sve_verdict", "Unknown"),
            inline=False
        )

        # ---- Source splitting ----
        if sources:
            MAX_FIELD_LEN = 1024
            chunk = ""

            for src in sources[:5]:
                title = src.get("title", "No title")
                link = src.get("link", "")
                entry = f"• [{title}]({link})\n"

                if len(chunk) + len(entry) > MAX_FIELD_LEN:
                    embed.add_field(
                        name="Top Sources",
                        value=chunk,
                        inline=False
                    )
                    chunk = entry
                else:
                    chunk += entry

            if chunk:
                embed.add_field(
                    name="Top Sources",
                    value=chunk,
                    inline=False
                )

        embed.set_footer(text="Aletheia - Restrict Misinformation, Empower Truth")

        await ctx.send(embed=embed)

    except Exception as e:
        await ctx.send(f"⚠️ Exception occurred: {str(e)}")



# -----------------------------
# MULTIMODAL (TEXT + IMAGE)
# -----------------------------
@bot.command()
async def predict_multimodal(ctx, *, text: str = None):
    """Predict news using text + image (attachment or reply)."""

    attachment = None

    # ---- Case 1: Image attached with command ----
    if ctx.message.attachments:
        attachment = ctx.message.attachments[0]

    # ---- Case 2: Reply to message containing image ----
    elif ctx.message.reference:
        ref_msg = await ctx.channel.fetch_message(
            ctx.message.reference.message_id
        )

        if ref_msg.attachments:
            attachment = ref_msg.attachments[0]

        # take replied message content as text if user didn't provide
        if not text and ref_msg.content:
            text = ref_msg.content

    if not attachment:
        await ctx.send(
            "⚠️ Attach an image or reply to an image message."
        )
        return

    if not text:
        await ctx.send("⚠️ Provide text claim as well.")
        return

    try:
        img_bytes = await attachment.read()

        files = {
            "image": (
                attachment.filename,
                BytesIO(img_bytes),
                attachment.content_type
            )
        }

        data_form = {"text": text}

        response = requests.post(
            f"{API_BASE_URL}/predict/multimodal",
            files=files,
            data=data_form
        )

        data = response.json()

        if "result" not in data:
            await ctx.send(
                f"⚠️ Error: {data.get('error', 'Unknown error')}"
            )
            return

        # API structure: result = [metrics, sources]
        result_data = data["result"][0]
        sources = data["result"][1]

        verdict = result_data.get("final_outcome", "Unknown")
        subtitle = result_data.get("subtitle", "")

        # ---- Color logic ----
        if "True" in verdict or "Authentic" in verdict:
            color = discord.Color.green()
        elif "Fake" in verdict or "Misleading" in verdict:
            color = discord.Color.red()
        else:
            color = discord.Color.orange()

        embed = discord.Embed(
            title=f"Verdict: {verdict}",
            description=subtitle,
            color=color
        )

        # ---- Metrics ----
        embed.add_field(
            name="Image Authenticity",
            value=result_data.get("image_integrity", "Unknown"),
            inline=True
        )

        embed.add_field(
            name="Image Confidence",
            value=f"{result_data.get('image_confidence', 0):.2f}%",
            inline=True
        )

        embed.add_field(
            name="Clickbait Confidence",
            value=f"{result_data.get('text_clickbait_confidence', 0):.2f}%",
            inline=True
        )

        embed.add_field(
            name="Text Claim Verdict",
            value=result_data.get("textual_claims", "Unknown"),
            inline=True
        )

        embed.add_field(
            name="Text-Image Match",
            value=result_data.get("text_image_match", "Unknown"),
            inline=True
        )

        embed.add_field(
            name="Sources Found",
            value=result_data.get("sources_found", 0),
            inline=True
        )

        embed.add_field(
            name="Trusted Sources",
            value=result_data.get("trusted_count", 0),
            inline=True
        )

        embed.add_field(
            name="Debunked Claims",
            value=result_data.get("debunk_count", 0),
            inline=True
        )

        # ---- Source splitting ----
        if sources:
            MAX_FIELD_LEN = 1024
            chunk = ""
            field_index = 1

            for src in sources:
                title = src.get("title", "No title")
                link = src.get("link", "")

                entry = f"• [{title}]({link})\n"

                if len(chunk) + len(entry) > MAX_FIELD_LEN:
                    embed.add_field(
                        name=f"Top Sources",
                        value=chunk,
                        inline=False
                    )
                    chunk = entry
                    field_index += 1
                else:
                    chunk += entry

            if chunk:
                embed.add_field(
                    name=f"Extra Sources",
                    value=chunk,
                    inline=False
                )

        embed.set_footer(text="Aletheia - Restrict Misinformation, Empower Truth")

        await ctx.send(embed=embed)

    except Exception as e:
        await ctx.send(f"⚠️ Exception occurred: {str(e)}")



# -----------------------------
# RUN BOT
# -----------------------------
bot.run(DISCORD_BOT_TOKEN)
