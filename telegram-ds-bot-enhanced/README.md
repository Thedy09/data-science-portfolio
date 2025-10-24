
# 🤖 Data Science Bot Enhanced 

A powerful, AI-powered Telegram bot that provides comprehensive data analysis, machine learning, and automated reporting capabilities. Upload your datasets and receive detailed insights, visualizations, and ML models in minutes!

## ✨ Features

### 🎯 **Core Capabilities**
- **📊 Data Analysis**: Upload CSV/XLSX files for instant analysis
- **🧹 Data Cleaning**: Automatic duplicate removal and missing value handling
- **📈 Exploratory Analysis**: Interactive HTML reports with rich visualizations
- **🤖 Machine Learning**: Multiple ML models with automatic comparison
- **💡 AI Insights**: GPT-powered analysis summaries and recommendations
- **📄 Multi-format Reports**: HTML, PDF, and comprehensive ZIP bundles

### 🚀 **Enhanced Features **
- **🎮 Interactive UI**: Button-based navigation and real-time progress tracking
- **⚙️ Customizable Settings**: Toggle ML models, AI summaries, and advanced statistics
- **📊 Advanced Statistics**: Normality tests, correlation analysis, outlier detection
- **🔍 Enhanced Visualizations**: Distribution plots, heatmaps, categorical analysis
- **🤖 Multiple ML Models**: Random Forest, Gradient Boosting, SVM, Linear models
- **📈 Model Comparison**: Visual performance comparison and cross-validation
- **⚡ Performance Monitoring**: Resource management and system health tracking
- **💾 Caching**: Intelligent caching for improved performance
- **📱 Rich Commands**: Comprehensive command system with help and status

### 🛠️ **Technical Features**
- **Async Processing**: Non-blocking analysis with progress indicators
- **Resource Management**: Automatic throttling and task management
- **Error Handling**: Comprehensive error handling with user-friendly messages
- **Performance Monitoring**: Real-time system metrics and health checks
- **Modular Architecture**: Clean, maintainable code structure

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Telegram Bot Token
- (Optional) OpenAI API Key for AI summaries

### Local Setup
1. **Clone and setup environment:**
   ```bash
   git clone <repository-url>
   cd telegram-ds-bot-enhanced
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env and add your tokens:
   # TELEGRAM_TOKEN=your_bot_token_here
   # OPENAI_API_KEY=your_openai_key_here (optional)
   # ENABLE_AUTOML=true (optional, for ML models)
   ```

3. **Run the bot:**
   ```bash
   python main.py
   ```

### Docker Setup
```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build and run manually
docker build -t ds-bot-enhanced .
docker run -d --env-file .env ds-bot-enhanced
```

## 📱 Usage

### Basic Commands
- `/start` - Welcome message and main menu
- `/help` - Comprehensive help and usage instructions
- `/settings` - Configure analysis preferences
- `/status` - Check current processing status
- `/history` - View analysis history
- `/system` - Show system status and performance metrics

### Analysis Workflow
1. **Upload Dataset**: Send a CSV or XLSX file as a document
2. **Progress Tracking**: Watch real-time progress updates
3. **Receive Results**: Get a comprehensive ZIP file with:
   - Interactive HTML report
   - PDF version of the report
   - Cleaned dataset (CSV)
   - ML models and metrics
   - Advanced visualizations
   - AI-powered insights
   - Statistical analysis

### Settings Configuration
Use `/settings` to customize your analysis:
- **🤖 ML Models**: Enable/disable machine learning model training
- **💡 AI Summaries**: Toggle GPT-powered insights
- **📊 Advanced Stats**: Include statistical tests and outlier detection

## 📊 Analysis Outputs

### 📄 **Reports**
- **Interactive HTML Report**: Navigate through sections, interact with visualizations
- **PDF Report**: Offline-friendly version for sharing
- **Summary Report**: Overview of all generated files and how to use them

### 📈 **Visualizations**
- **Distribution Analysis**: Histograms, box plots, correlation heatmaps
- **Categorical Analysis**: Bar charts for categorical variables
- **Model Comparison**: Performance metrics visualization
- **Missing Data Analysis**: Visual representation of data completeness

### 🤖 **Machine Learning**
- **Multiple Models**: Random Forest, Gradient Boosting, SVM, Linear models
- **Automatic Selection**: Best model chosen based on performance
- **Cross-Validation**: Robust performance evaluation
- **Model Artifacts**: Saved models for future use

### 📊 **Statistics**
- **Descriptive Statistics**: Comprehensive data summaries
- **Normality Tests**: Shapiro-Wilk tests for numeric variables
- **Correlation Analysis**: Pearson, Spearman, and Kendall correlations
- **Outlier Detection**: IQR-based outlier identification

## ⚙️ Configuration

### Environment Variables
```bash
# Required
TELEGRAM_TOKEN=your_bot_token_here

# Optional
OPENAI_API_KEY=your_openai_key_here  # For AI summaries
ENABLE_AUTOML=true                   # Enable ML model training
MAX_FILE_MB=20                       # Maximum file size in MB
```

### Bot Settings
- **File Size Limit**: 20MB (configurable)
- **Supported Formats**: CSV, XLSX, XLS
- **Concurrent Analyses**: 3 (configurable)
- **Cache TTL**: 1 hour (configurable)

## 🏗️ Architecture

### Core Modules
- **`main.py`**: Bot interface and command handling
- **`data_processor.py`**: Data cleaning and analysis
- **`model_builder.py`**: Machine learning model training
- **`report_generator.py`**: Report generation and bundling
- **`gpt_summary.py`**: AI-powered insights
- **`performance.py`**: Performance monitoring and resource management

### Key Features
- **Async Processing**: Non-blocking analysis with progress tracking
- **Resource Management**: Automatic throttling and task management
- **Error Handling**: Comprehensive error handling with user feedback
- **Caching**: Intelligent caching for improved performance
- **Monitoring**: Real-time system health and performance metrics

## 🚀 Deployment

### Railway
1. Connect your GitHub repository
2. Set environment variables
3. Deploy automatically

### Docker
```bash
# Build image
docker build -t ds-bot-enhanced .

# Run container
docker run -d --env-file .env ds-bot-enhanced
```

### Heroku
1. Create a new Heroku app
2. Set environment variables
3. Deploy using Git or GitHub integration

## 📈 Performance

### System Requirements
- **Memory**: 512MB minimum, 1GB recommended
- **CPU**: 1 core minimum, 2 cores recommended
- **Storage**: 1GB for temporary files
- **Network**: Stable internet connection

### Optimization Features
- **Resource Monitoring**: Automatic throttling when resources are high
- **Task Management**: Queue management for concurrent analyses
- **Memory Optimization**: Automatic garbage collection
- **Caching**: Intelligent caching for repeated operations

## 🔧 Troubleshooting

### Common Issues
1. **File too large**: Reduce file size or increase `MAX_FILE_MB`
2. **Analysis fails**: Check file format and data quality
3. **High memory usage**: Enable resource throttling
4. **Slow processing**: Check system resources with `/system`

### Debug Commands
- `/system` - Check system health and performance
- `/status` - View current processing status
- Check logs for detailed error information

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning algorithms
- **matplotlib/seaborn**: Data visualization
- **ydata-profiling**: Automated EDA reports
- **python-telegram-bot**: Telegram bot framework
- **OpenAI**: AI-powered insights

---

**Made with ❤️ for the data science community**

*For support or questions, please open an issue or contact the maintainers.*

# data-science-portfolio
My journey in data science 

