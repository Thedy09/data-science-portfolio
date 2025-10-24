# main.py
import os
import tempfile
import logging
import random
import asyncio
import json
import time
from pathlib import Path
from datetime import datetime
from telegram import Update, InputFile, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
from telegram.constants import ParseMode
from dotenv import load_dotenv

from data_processor import process_dataset
from report_generator import bundle_outputs
from gpt_summary import generate_gpt_summary
from performance import (monitor_performance, resource_manager, perf_monitor, 
                     cache_manager, check_system_health, timeout_manager)

load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
MAX_FILE_MB = int(os.getenv("MAX_FILE_MB", "20")) * 1024 * 1024

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# User session storage
user_sessions = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_sessions[user_id] = {
        'processing': False,
        'last_analysis': None,
        'preferences': {'include_ml': True, 'include_gpt': True}
    }
    
    welcome_text = """
🤖 **Welcome to Thedysci Bot!**

I'm your AI-powered data analysis assistant. I can help you:

📊 **Analyze your datasets** - Upload CSV/XLSX files
🔍 **Generate insights** - Automatic EDA and statistical analysis  
🤖 **Build ML models** - Classification and regression models
📈 **Create reports** - Beautiful HTML/PDF reports with visualizations
💡 **AI summaries** - GPT-powered insights and recommendations

**How to use:**
1. Send me a CSV or XLSX file as a document
2. I'll analyze it and send back a comprehensive report
3. Use /settings to customize your analysis preferences

Ready to analyze some data? Just send me a file! 🚀
"""
    
    keyboard = [
        [InlineKeyboardButton("📊 Upload Dataset", callback_data="upload")],
        [InlineKeyboardButton("⚙️ Settings", callback_data="settings"),
         InlineKeyboardButton("❓ Help", callback_data="help")],
        [InlineKeyboardButton("📈 Sample Analysis", callback_data="sample")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(welcome_text, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup)

def _is_allowed(filename: str):
    return any(filename.lower().endswith(ext) for ext in (".csv", ".xlsx", ".xls"))

@monitor_performance
async def handle_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    logger.info(f"handle_file called for user={user_id}")
    
    # Check if we can start a new analysis with enhanced reason
    can_start, reason = await resource_manager.can_start_analysis(user_id)
    if not can_start:
        await update.message.reply_text(f"⏳ Cannot start analysis: {reason}\nPlease try again in a few minutes.")
    
    # Check if user is already processing
    if user_id in user_sessions and user_sessions[user_id].get('processing', False):
        await update.message.reply_text("⏳ You already have an analysis in progress. Please wait for it to complete.")
        return
    
    msg = update.message
    if not msg or not msg.document:
        await update.message.reply_text("❌ Please send a file as a document.")
        return

    doc = msg.document
    filename = doc.file_name or "dataset"
    
    # Enhanced file validation
    if doc.file_size and doc.file_size > MAX_FILE_MB:
        await msg.reply_text(f"❌ File too large! Maximum size is {MAX_FILE_MB // (1024*1024)} MB.\nYour file: {doc.file_size // (1024*1024)} MB")
        return

    if not _is_allowed(filename):
        supported_formats = "`.csv`, `.xlsx`, `.xls`"
        await msg.reply_text(f"❌ Unsupported file type!\n\nSupported formats: {supported_formats}\nYour file: `{filename}`", parse_mode=ParseMode.MARKDOWN)
        return

    # Set processing status
    if user_id not in user_sessions:
        user_sessions[user_id] = {'processing': False, 'preferences': {'include_ml': True, 'include_gpt': True}}
    user_sessions[user_id]['processing'] = True
    logger.info(f"User {user_id} session set to processing")
    
    # Register analysis start
    task_id = f"{user_id}_{int(time.time())}"
    resource_manager.start_analysis(user_id, task_id)

    # Acknowledge quickly and run processing in background
    progress_msg = await msg.reply_text("📁 **File received!** Starting analysis in background...")
    logger.info(f"User {user_id}: acknowledged and scheduling background worker")

    async def _worker():
        """Background worker that performs processing and messaging."""
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                local_path = Path(tmpdir) / filename
                file = await doc.get_file()
                await file.download_to_drive(custom_path=str(local_path))
                await progress_msg.edit_text("✅ Step 1/5: File downloaded\n⏳ Step 2/5: Processing data...")

                # Process dataset
                try:
                    outputs = await timeout_manager.run_with_timeout(
                        process_dataset,
                        str(local_path),
                        tmpdir,
                        filename,
                        user_sessions[user_id]['preferences']
                    )
                except asyncio.TimeoutError:
                    await progress_msg.edit_text("❌ Processing timed out. Try a smaller file.")
                    return
                except Exception as e:
                    await progress_msg.edit_text(f"❌ Error during processing: {e}")
                    return

                await progress_msg.edit_text("✅ Step 2/5: Data processed\n⏳ Step 3/5: Generating reports...")

                # GPT summary (optional)
                if user_sessions[user_id]['preferences'].get('include_gpt', True):
                    await progress_msg.edit_text("⏳ Step 4/5: AI analysis (may be skipped if API limits)")
                    cache_key = f"gpt_summary_{hash(str(outputs.get('summary_text', '')))}"
                    summary_text = cache_manager.get(cache_key)
                    if not summary_text:
                        try:
                            summary_text = await timeout_manager.run_with_timeout(
                                generate_gpt_summary,
                                outputs.get('summary_text', ''),
                                timeout=60
                            )
                            cache_manager.set(cache_key, summary_text)
                        except Exception as e:
                            logger.warning(f"GPT summary skipped/failed: {e}")
                            summary_text = None

                    if summary_text:
                        summary_file = Path(tmpdir) / "gpt_summary.txt"
                        summary_file.write_text(summary_text, encoding='utf-8')
                        outputs['files'].append(str(summary_file))

                await progress_msg.edit_text("⏳ Step 5/5: Creating bundle...")

                # Bundle outputs with improved logging and fallback
                zip_path = Path(tmpdir) / "report_bundle.zip"
                try:
                    # Log files to be bundled
                    file_list = outputs.get('files', []) if isinstance(outputs, dict) else outputs
                    logger.info(f"Creating bundle for user {user_id}, {len(file_list)} files: {file_list}")

                    # Increase timeout for bundling (some reports can be large)
                    top_dir = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    await timeout_manager.run_with_timeout(
                        bundle_outputs,
                        file_list,
                        str(zip_path),
                        top_dir,
                        timeout=180
                    )

                    # Verify ZIP was created
                    if not zip_path.exists() or zip_path.stat().st_size == 0:
                        raise RuntimeError("Bundle file not created or empty")

                    await progress_msg.edit_text("✅ Bundle created, sending results...")
                    await update.message.reply_document(
                        document=InputFile(open(zip_path, 'rb')),
                        filename=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
                    )

                except asyncio.TimeoutError:
                    logger.exception(f"Bundle creation timed out for user {user_id}")
                    await progress_msg.edit_text("⚠️ Bundling timed out. Sending individual files instead...")
                    # Fallback: send individual files (skip non-existing)
                    for fp in (outputs.get('files', []) if isinstance(outputs, dict) else outputs):
                        try:
                            if Path(fp).exists():
                                await update.message.reply_document(document=InputFile(open(fp, 'rb')))
                        except Exception as exc:
                            logger.warning(f"Failed to send file {fp}: {exc}")
                    await progress_msg.edit_text("✅ Sent available files (bundling timed out)")
                except Exception as e:
                    logger.exception(f"Bundle creation failed for user {user_id}: {e}")
                    await progress_msg.edit_text(f"❌ Bundling failed: {e}\nAttempting to send available files...")
                    # Try to send individual files as a fallback
                    sent_any = False
                    for fp in (outputs.get('files', []) if isinstance(outputs, dict) else outputs):
                        try:
                            if Path(fp).exists():
                                await update.message.reply_document(document=InputFile(open(fp, 'rb')))
                                sent_any = True
                        except Exception as exc:
                            logger.warning(f"Failed to send file {fp}: {exc}")
                    if sent_any:
                        await progress_msg.edit_text("✅ Sent available files (bundle failed)")
                    else:
                        await progress_msg.edit_text("❌ No files available to send after bundle failure")

                # Store analysis info
                user_sessions[user_id]['last_analysis'] = {
                    'filename': filename,
                    'timestamp': datetime.now().isoformat(),
                    'file_count': len(outputs['files'])
                }

        finally:
            # Reset processing status and end analysis
            if user_id in user_sessions:
                user_sessions[user_id]['processing'] = False
            resource_manager.end_analysis(user_id)

    # Start background worker; don't await it here to keep handler responsive
    asyncio.create_task(_worker())
    return

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = """
🤖 **Thedsci Bot- Help**

**📋 Commands:**
• `/start` - Welcome message and main menu
• `/help` - Show this help message
• `/settings` - Configure analysis preferences
• `/status` - Check processing status
• `/history` - View your analysis history

**📊 How to use:**
1. Send me a CSV or XLSX file as a document
2. I'll automatically analyze it and generate reports
3. Receive a ZIP file with comprehensive results

**🔧 Features:**
• **Data Cleaning** - Automatic duplicate removal and missing value handling
• **Exploratory Analysis** - Interactive HTML reports with visualizations
• **Machine Learning** - Optional ML model training and evaluation
• **AI Insights** - GPT-powered analysis summaries and recommendations
• **Multiple Formats** - HTML, PDF, and CSV outputs

**📁 Supported Files:**
• CSV files (`.csv`)
• Excel files (`.xlsx`, `.xls`)
• Maximum size: 20MB

**⚙️ Customization:**
Use `/settings` to enable/disable:
• Machine learning model training
• AI-powered summaries
• Advanced statistical tests

Need more help? Just ask! 🚀
"""
    await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)

async def settings_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in user_sessions:
        user_sessions[user_id] = {'processing': False, 'preferences': {'include_ml': True, 'include_gpt': True}}
    
    prefs = user_sessions[user_id]['preferences']
    
    settings_text = f"""
⚙️ **Settings & Preferences**

**Current Settings:**
• 🤖 Machine Learning: {'✅ Enabled' if prefs.get('include_ml', True) else '❌ Disabled'}
• 💡 AI Summaries: {'✅ Enabled' if prefs.get('include_gpt', True) else '❌ Disabled'}
• 📊 Advanced Stats: {'✅ Enabled' if prefs.get('include_advanced_stats', True) else '❌ Disabled'}

**What each setting does:**
• **ML Models** - Train classification/regression models on your data
• **AI Summaries** - Generate GPT-powered insights and recommendations
• **Advanced Stats** - Include correlation analysis, statistical tests, and outlier detection

Use the buttons below to toggle settings:
"""
    
    keyboard = [
        [InlineKeyboardButton(
            f"🤖 ML Models: {'ON' if prefs.get('include_ml', True) else 'OFF'}",
            callback_data="toggle_ml"
        )],
        [InlineKeyboardButton(
            f"💡 AI Summaries: {'ON' if prefs.get('include_gpt', True) else 'OFF'}",
            callback_data="toggle_gpt"
        )],
        [InlineKeyboardButton(
            f"📊 Advanced Stats: {'ON' if prefs.get('include_advanced_stats', True) else 'OFF'}",
            callback_data="toggle_stats"
        )],
        [InlineKeyboardButton("🔙 Back to Main Menu", callback_data="main_menu")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(settings_text, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup)

async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    
    if user_id not in user_sessions:
        await update.message.reply_text("❌ No session found. Use /start to begin!")
        return
    
    session = user_sessions[user_id]
    
    if session.get('processing', False):
        status_text = "⏳ **Processing Status:** Currently analyzing a dataset...\n\nPlease wait for the current analysis to complete."
    else:
        status_text = "✅ **Processing Status:** Ready for new analysis\n\n"
        
        if session.get('last_analysis'):
            last = session['last_analysis']
            status_text += f"**Last Analysis:**\n"
            status_text += f"• File: `{last['filename']}`\n"
            status_text += f"• Time: {last['timestamp']}\n"
            status_text += f"• Files generated: {last['file_count']}\n"
        else:
            status_text += "No previous analyses found."
    
    await update.message.reply_text(status_text, parse_mode=ParseMode.MARKDOWN)

async def history_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    
    if user_id not in user_sessions or not user_sessions[user_id].get('last_analysis'):
        await update.message.reply_text("📊 **Analysis History:** No previous analyses found.\n\nUpload a dataset to get started!")
        return
    
    last = user_sessions[user_id]['last_analysis']
    history_text = f"""
📊 **Analysis History**

**Most Recent Analysis:**
• 📁 **File:** `{last['filename']}`
• 🕒 **Date:** {last['timestamp']}
• 📈 **Files Generated:** {last['file_count']}

**Note:** Currently only showing the most recent analysis. Full history tracking coming soon!
"""
    
    await update.message.reply_text(history_text, parse_mode=ParseMode.MARKDOWN)

async def system_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show system status and performance metrics"""
    try:
        health = await check_system_health()
        
        status_emoji = {
            'healthy': '✅',
            'warning': '⚠️',
            'critical': '🚨'
        }
        
        status_text = f"""
{status_emoji.get(health['status'], '❓')} **System Status: {health['status'].upper()}**

**📊 Performance Metrics:**
• 🧠 Memory: {health['performance']['memory_percent']:.1f}% ({health['performance']['memory_trend']})
• ⚡ CPU: {health['performance']['cpu_percent']:.1f}% ({health['performance']['cpu_trend']})
• 💾 Disk: {health['performance']['disk_percent']:.1f}%
• ⏱️ Uptime: {health['performance']['uptime']//3600:.0f}h {(health['performance']['uptime']%3600)//60:.0f}m

**📈 Analysis Statistics:**
• 📊 Total Analyses: {health['performance']['total_analyses']}
• ✅ Success Rate: {health['performance']['success_rate']:.1%}
• ⏱️ Avg Time: {health['performance']['avg_processing_time']:.1f}s
• ⚠️ Timeout Rate: {health['performance']['timeout_rate']:.1%}

**🔄 Resource Status:**
• 🔄 Active: {health['resources']['active_tasks']}/{health['resources']['max_concurrent']}
• ⏱️ Task Ages: {', '.join(f'{age:.0f}s' for age in health['resources']['task_ages'].values()) if health['resources']['task_ages'] else 'None'}
• 🆕 Can Accept: {'✅' if health['resources']['can_accept_new'] else '❌'}

**💾 Cache Status:**
• 📦 Size: {health['cache']['size']}/{health['cache']['max_size']}
• 🎯 Hit Rate: {health['cache']['hit_rate']:.1%}
• ⏰ TTL: {health['cache']['ttl']}s

**🔧 System Recommendations:**
{chr(10).join(f'• {r}' for r in health['recommendations'])}

*Last updated: {health['timestamp']}*
"""
        
        await update.message.reply_text(status_text, parse_mode=ParseMode.MARKDOWN)
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        await update.message.reply_text("❌ **Error retrieving system status.**\n\nPlease try again later.")

async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    user_id = query.from_user.id
    data = query.data
    
    if data == "upload":
        await query.edit_message_text("📊 **Upload a Dataset**\n\nSend me a CSV or XLSX file as a document to analyze it!\n\nSupported formats: `.csv`, `.xlsx`, `.xls`\nMaximum size: 20MB", parse_mode=ParseMode.MARKDOWN)
    
    elif data == "settings":
        if user_id not in user_sessions:
            user_sessions[user_id] = {'processing': False, 'preferences': {'include_ml': True, 'include_gpt': True}}
        
        prefs = user_sessions[user_id]['preferences']
        
        settings_text = f"""
⚙️ **Settings & Preferences**

**Current Settings:**
• 🤖 Machine Learning: {'✅ Enabled' if prefs.get('include_ml', True) else '❌ Disabled'}
• 💡 AI Summaries: {'✅ Enabled' if prefs.get('include_gpt', True) else '❌ Disabled'}
• 📊 Advanced Stats: {'✅ Enabled' if prefs.get('include_advanced_stats', True) else '❌ Disabled'}

Use the buttons below to toggle settings:
"""
        
        keyboard = [
            [InlineKeyboardButton(
                f"🤖 ML Models: {'ON' if prefs.get('include_ml', True) else 'OFF'}", 
                callback_data="toggle_ml"
            )],
            [InlineKeyboardButton(
                f"💡 AI Summaries: {'ON' if prefs.get('include_gpt', True) else 'OFF'}", 
                callback_data="toggle_gpt"
            )],
            [InlineKeyboardButton(
                f"📊 Advanced Stats: {'ON' if prefs.get('include_advanced_stats', True) else 'OFF'}", 
                callback_data="toggle_stats"
            )],
            [InlineKeyboardButton("🔙 Back to Main Menu", callback_data="main_menu")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(settings_text, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup)
    
    elif data == "help":
        help_text = """
🤖 **Thedsci Bot - Help**

**📋 Commands:**
• `/start` - Welcome message and main menu
• `/help` - Show this help message
• `/settings` - Configure analysis preferences
• `/status` - Check processing status
• `/history` - View your analysis history

**📊 How to use:**
1. Send me a CSV or XLSX file as a document
2. I'll automatically analyze it and generate reports
3. Receive a ZIP file with comprehensive results

**🔧 Features:**
• **Data Cleaning** - Automatic duplicate removal and missing value handling
• **Exploratory Analysis** - Interactive HTML reports with visualizations
• **Machine Learning** - Optional ML model training and evaluation
• **AI Insights** - GPT-powered analysis summaries and recommendations
• **Multiple Formats** - HTML, PDF, and CSV outputs

**📁 Supported Files:**
• CSV files (`.csv`)
• Excel files (`.xlsx`, `.xls`)
• Maximum size: 20MB

Need more help? Just ask! 🚀
"""
        keyboard = [[InlineKeyboardButton("🔙 Back to Main Menu", callback_data="main_menu")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(help_text, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup)
    
    elif data == "sample":
        sample_text = """
📈 **Sample Analysis**

Here's what a typical analysis includes:

**📊 Data Overview:**
• Dataset shape and basic statistics
• Data types and missing value analysis
• Duplicate detection and removal

**🔍 Exploratory Analysis:**
• Interactive visualizations (histograms, scatter plots, heatmaps)
• Correlation analysis between variables
• Outlier detection and analysis
• Distribution analysis

**🤖 Machine Learning (if enabled):**
• Automatic model selection (classification/regression)
• Model performance metrics
• Feature importance analysis
• Model artifacts for future use

**💡 AI Insights (if enabled):**
• Natural language summary of findings
• Actionable recommendations
• Data quality assessment
• Next steps suggestions

**📁 Output Files:**
• Interactive HTML report
• PDF version of the report
• Cleaned dataset (CSV)
• Model files and metrics
• AI summary document

Ready to analyze your own data? Send me a file! 🚀
"""
        keyboard = [[InlineKeyboardButton("🔙 Back to Main Menu", callback_data="main_menu")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(sample_text, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup)
    
    elif data == "main_menu":
        welcome_text = """
🤖 **Welcome to Data Science Bot Enhanced!**

I'm your AI-powered data analysis assistant. I can help you:

📊 **Analyze your datasets** - Upload CSV/XLSX files
🔍 **Generate insights** - Automatic EDA and statistical analysis  
🤖 **Build ML models** - Classification and regression models
📈 **Create reports** - Beautiful HTML/PDF reports with visualizations
💡 **AI summaries** - GPT-powered insights and recommendations

**How to use:**
1. Send me a CSV or XLSX file as a document
2. I'll analyze it and send back a comprehensive report
3. Use /settings to customize your analysis preferences

Ready to analyze some data? Just send me a file! 🚀
"""
        
        keyboard = [
            [InlineKeyboardButton("📊 Upload Dataset", callback_data="upload")],
            [InlineKeyboardButton("⚙️ Settings", callback_data="settings"),
             InlineKeyboardButton("❓ Help", callback_data="help")],
            [InlineKeyboardButton("📈 Sample Analysis", callback_data="sample")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(welcome_text, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup)
    
    elif data.startswith("toggle_"):
        if user_id not in user_sessions:
            user_sessions[user_id] = {'processing': False, 'preferences': {'include_ml': True, 'include_gpt': True}}
        
        setting = data.replace("toggle_", "")
        current_value = user_sessions[user_id]['preferences'].get(f'include_{setting}', True)
        user_sessions[user_id]['preferences'][f'include_{setting}'] = not current_value
        
        # Show updated settings
        prefs = user_sessions[user_id]['preferences']
        
        settings_text = f"""
⚙️ **Settings & Preferences**

**Current Settings:**
• 🤖 Machine Learning: {'✅ Enabled' if prefs.get('include_ml', True) else '❌ Disabled'}
• 💡 AI Summaries: {'✅ Enabled' if prefs.get('include_gpt', True) else '❌ Disabled'}
• 📊 Advanced Stats: {'✅ Enabled' if prefs.get('include_advanced_stats', True) else '❌ Disabled'}

Use the buttons below to toggle settings:
"""
        
        keyboard = [
            [InlineKeyboardButton(
                f"🤖 ML Models: {'ON' if prefs.get('include_ml', True) else 'OFF'}", 
                callback_data="toggle_ml"
            )],
            [InlineKeyboardButton(
                f"💡 AI Summaries: {'ON' if prefs.get('include_gpt', True) else 'OFF'}", 
                callback_data="toggle_gpt"
            )],
            [InlineKeyboardButton(
                f"📊 Advanced Stats: {'ON' if prefs.get('include_advanced_stats', True) else 'OFF'}", 
                callback_data="toggle_stats"
            )],
            [InlineKeyboardButton("🔙 Back to Main Menu", callback_data="main_menu")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(settings_text, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup)


async def post_init(application):
    """Set bot commands after initialization"""
    try:
        commands = [
            BotCommand("start", "🚀 Start the bot and show main menu"),
            BotCommand("help", "❓ Show help and usage instructions"),
            BotCommand("settings", "⚙️ Configure analysis preferences"),
            BotCommand("status", "📊 Check current processing status"),
            BotCommand("history", "📈 View analysis history"),
            BotCommand("system", "🔧 Show system status and performance")
        ]
        await application.bot.set_my_commands(commands)
        logger.info("✅ Bot commands configured successfully")
    except Exception as e:
        logger.error(f"❌ Error setting bot commands: {e}")

def main():
    try:
        if not TELEGRAM_TOKEN:
            raise RuntimeError('TELEGRAM_TOKEN not set in env')
        
        logger.info("🔧 Building application...")
        app = ApplicationBuilder().token(TELEGRAM_TOKEN).post_init(post_init).build()
        
        logger.info("📝 Adding command handlers...")
        # Command handlers
        app.add_handler(CommandHandler('start', start))
        app.add_handler(CommandHandler('help', help_cmd))
        app.add_handler(CommandHandler('settings', settings_cmd))
        app.add_handler(CommandHandler('status', status_cmd))
        app.add_handler(CommandHandler('history', history_cmd))
        app.add_handler(CommandHandler('system', system_cmd))
        
        # Callback query handler for interactive buttons
        app.add_handler(CallbackQueryHandler(callback_handler))
        
        # Message handlers
        app.add_handler(MessageHandler(filters.Document.ALL & ~filters.COMMAND, handle_file))
        
        logger.info("🤖 Data Science Bot Enhanced starting (polling mode)...")
        logger.info("📊 Features: Interactive UI, Progress tracking, Settings, ML models, AI summaries")
        logger.info("🚀 Starting polling...")
        
        app.run_polling()
        
    except Exception as e:
        logger.error(f"❌ Fatal error in main: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == '__main__':
    main()
