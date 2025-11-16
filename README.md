
import logging
import io
import os
import random
import matplotlib.pyplot as plt
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InputFile
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext, CallbackQueryHandler

# --- إعدادات أساسية ---
# يقرأ التوكن من متغيرات البيئة لضمان الأمان
TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")

# إعداد تسجيل الدخول لعرض الأخطاء
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- تحميل النماذج عند بدء التشغيل ---
# تم استخدام نموذج أصغر ليتناسب مع الخوادم المجانية (مثل Render Free Tier)
AI_DETECTOR_MODEL = 'distilroberta-base-openai-detector'
ai_detector_pipeline = None

def load_models():
    """تحميل نماذج الذكاء الاصطناعي عند الحاجة."""
    global ai_detector_pipeline
    if ai_detector_pipeline is None:
        try:
            from transformers import pipeline
            logger.info(f"جاري تحميل نموذج الكشف: {AI_DETECTOR_MODEL}...")
            ai_detector_pipeline = pipeline('text-classification', model=AI_DETECTOR_MODEL)
            logger.info("تم تحميل النموذج بنجاح.")
        except ImportError:
            logger.error("مكتبة 'transformers' غير مثبتة. خدمة الكشف لن تعمل.")
        except Exception as e:
            logger.error(f"فشل تحميل نموذج الكشف (قد تكون مشكلة ذاكرة): {e}")

# --- الوظائف الأساسية ---

def latex_to_image_bytes(latex_string: str):
    """تحويل سلسلة LaTeX إلى صورة PNG كـ bytes."""
    try:
        full_latex_str = f"${latex_string}$"
        fig, ax = plt.subplots(figsize=(5, 1), dpi=300)
        ax.axis('off')
        ax.text(0.5, 0.5, full_latex_str, size=15, ha='center', va='center')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.2, transparent=True)
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        logger.error(f"خطأ في تحويل LaTeX: {e}")
        return None

def check_ai_text(text_to_check: str):
    """يستخدم نموذجاً مدرباً لتقدير احتمالية أن يكون النص مولداً بالذكاء الاصطناعي."""
    load_models() # التأكد من تحميل النموذج
    if not ai_detector_pipeline:
        return "عذراً، خدمة كشف النصوص غير متاحة حالياً بسبب مشكلة في تحميل النموذج."
    try:
        results = ai_detector_pipeline(text_to_check)
        # تحديد النتيجة بناءً على التسمية (label)
        # 'LABEL_1' أو 'Real' عادةً للنص البشري، 'LABEL_0' أو 'Fake' للآلي
        ai_score = results[0]['score'] if results[0]['label'].upper() in ['FAKE', 'LABEL_0'] else 1 - results[0]['score']
        
        if ai_score > 0.8:
            return f"🚨 **تم الكشف بنسبة عالية ({ai_score:.0%}) أن هذا النص مولّد بواسطة AI.**"
        elif ai_score > 0.5:
            return f"⚠️ **هناك احتمال ({ai_score:.0%}) أن هذا النص مولّد بواسطة AI.**"
        else:
            return f"✅ **على الأرجح، هذا النص مكتوب بواسطة إنسان.** (احتمال AI: {ai_score:.0%})"
    except Exception as e:
        logger.error(f"خطأ في فحص النص: {e}")
        return "حدث خطأ أثناء محاولة تحليل النص."

def humanize_text(ai_text: str) -> str:
    """تعديل نص مولّد آلياً ليبدو أكثر طبيعية وبشرية."""
    sentences = ai_text.split('. ')
    new_sentences = []
    common_ai_phrases = ["في الختام،", "يمكن القول أن", "من ناحية أخرى،", "علاوة على ذلك،", "في نهاية المطاف،"]
    for i, sentence in enumerate(sentences):
        for phrase in common_ai_phrases:
            sentence = sentence.replace(phrase, "").strip()
        if random.random() < 0.15: # تقليل احتمالية إضافة كلمات لتجنب التكرار
            prefix = random.choice(["في الواقع، ", "في رأيي، ", "بصراحة، "])
            sentence = prefix + sentence[0].lower() + sentence[1:] if sentence else ""
        new_sentences.append(sentence.strip())
    humanized_output = ". ".join(filter(None, new_sentences))
    # التأكد من أن الحرف الأول كبير
    if humanized_output:
        humanized_output = humanized_output[0].upper() + humanized_output[1:]
    return ' '.join(humanized_output.split())

# --- معالجات أوامر البوت ---

async def start(update: Update, context: CallbackContext) -> None:
    """إرسال الرسالة الترحيبية مع الأزرار."""
    keyboard = [
        [InlineKeyboardButton("🔍 كشف نص AI", callback_data='detect_ai')],
        [InlineKeyboardButton("✍️ تحويل النص لبشري", callback_data='humanize_text')],
        [InlineKeyboardButton("🖼️ تحويل LaTeX لصورة", callback_data='latex_to_image')],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    welcome_message = "أهلاً بك في بوت المعالجة المتقدمة للنصوص!\n\nاختر إحدى الخدمات من القائمة أدناه:"
    
    # تحديد ما إذا كان يجب إرسال رسالة جديدة أو تعديل رسالة موجودة
    if update.callback_query:
        await update.callback_query.edit_message_text(welcome_message, reply_markup=reply_markup, parse_mode='Markdown')
    else:
        await update.message.reply_text(welcome_message, reply_markup=reply_markup, parse_mode='Markdown')

async def button(update: Update, context: CallbackContext) -> None:
    """معالجة الضغط على الأزرار."""
    query = update.callback_query
    await query.answer()
    context.user_data['choice'] = query.data
    
    prompts = {
        'detect_ai': "الآن، أرسل لي النص أو ملف `.txt` الذي تريد فحصه.",
        'humanize_text': "الآن، أرسل لي النص أو ملف `.txt` الذي تريد 'أنسنته'.",
        'latex_to_image': "الآن، أرسل لي معادلة LaTeX (بدون $).\nمثال: `\\frac{a^2}{b_i}`"
    }
    await query.edit_message_text(text=prompts[query.data], parse_mode='Markdown')

async def handle_text_or_file(update: Update, context: CallbackContext) -> None:
    """معالجة الرسائل النصية والملفات."""
    if 'choice' not in context.user_data:
        await update.message.reply_text("الرجاء اختيار خدمة أولاً من خلال الأمر /start.")
        return

    user_choice = context.user_data['choice']
    user_text = ""

    if update.message.text:
        user_text = update.message.text
    elif update.message.document:
        if update.message.document.mime_type == 'text/plain':
            file = await update.message.document.get_file()
            file_bytes = await file.download_as_bytearray()
            user_text = file_bytes.decode('utf-8')
        else:
            await update.message.reply_text("الملف غير مدعوم. الرجاء إرسال ملف نصي (`.txt`).")
            return
    
    if not user_text:
        return

    processing_message = await update.message.reply_text("⏳ ...جاري المعالجة، يرجى الانتظار...", parse_mode='Markdown')
    
    try:
        if user_choice == 'detect_ai':
            result = check_ai_text(user_text)
            await processing_message.edit_text(result, parse_mode='Markdown')
        elif user_choice == 'humanize_text':
            result = humanize_text(user_text)
            await processing_message.edit_text(result)
        elif user_choice == 'latex_to_image':
            image_bytes = latex_to_image_bytes(user_text)
            if image_bytes:
                await update.message.reply_photo(photo=image_bytes, caption=f"صورة المعادلة:\n`{user_text}`", parse_mode='Markdown')
                await processing_message.delete()
            else:
                await processing_message.edit_text("حدث خطأ أثناء تحويل المعادلة. تأكد من صحة صيغة LaTeX.")
    except Exception as e:
        logger.error(f"خطأ كبير في المعالجة: {e}")
        await processing_message.edit_text("عذراً، حدث خطأ غير متوقع. يرجى المحاولة مرة أخرى.")

    # إعادة تعيين الحالة والعودة إلى القائمة الرئيسية
    if 'choice' in context.user_data:
        del context.user_data['choice']
    await start(update, context)

def main() -> None:
    """تشغيل البوت."""
    if not TOKEN:
        logger.critical("لم يتم العثور على توكن البوت! الرجاء تعيين متغير البيئة TELEGRAM_BOT_TOKEN في منصة الاستضافة.")
        return
        
    logger.info("بدء تشغيل البوت...")
    application = Application.builder().token(TOKEN).build()
    
    # إضافة المعالجات
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CallbackQueryHandler(button))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_or_file))
    application.add_handler(MessageHandler(filters.Document.MimeType('text/plain'), handle_text_or_file))
    
    # بدء تحميل النماذج في الخلفية
    load_models()

    # تشغيل البوت
    application.run_polling()
    logger.info("توقف البوت.")

if __name__ == '__main__':
    main()
    
