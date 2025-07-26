import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title=" Linear Regression Calculator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
body {
    background-color: #121212; /* consistent dark background */
    color: #f0f0f0; /* default text color */
}

.main-header {
    font-size: 3rem;
    color: #90cdf4; /* soft blue suitable for dark mode */
    text-align: center;
    margin-bottom: 2rem;
    font-weight: bold;
}

.metric-card {
    background-color: #1e1e1e; /* darker card background */
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
    border-left: 4px solid #63b3ed; /* soft cyan-blue border */
    color: #ffffff; /* ensure readable text */
}

.stMetric {
    background-color: #2a2a2a; /* dark background for metrics */
    border: 1px solid #3a3a3a;
    padding: 1rem;
    border-radius: 8px;
    color: #ffffff; /* white text for visibility */
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.5);
}

.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
}

.stTabs [data-baseweb="tab"] {
    height: 50px;
    white-space: pre-wrap;
    background-color: #2d2d2d;
    border-radius: 5px 5px 0 0;
    color: #ccc;
    padding-left: 20px;
    padding-right: 20px;
}

.stTabs [aria-selected="true"] {
    background-color: #3182ce; /* a nice bright blue for selected tab */
    color: white;
    border-bottom: 3px solid #3182ce;
}

.stTabs [data-baseweb="tab"]:hover {
    background-color: #3a3a3a;
    color: #f0f0f0;
    cursor: pointer;
}

</style>
""", unsafe_allow_html=True)

def calculate_regression(x, y, lr, iterations=1):
    """Enhanced linear regression with multiple iterations and  metrics"""
    
    x = np.array(x)
    y = np.array(y)
    n = len(x)
    
    # Calculate basic statistics
    meanofx = np.mean(x)
    meanofy = np.mean(y)
    std_x = np.std(x)
    std_y = np.std(y)
    
    # Calculate correlation coefficient
    correlation = np.corrcoef(x, y)[0, 1]
    
    # Least squares solution
    numerator = np.sum((x - meanofx) * (y - meanofy))
    denominator = np.sum((x - meanofx) ** 2)
    
    slope = numerator / denominator
    intercept = meanofy - slope * meanofx
    
    # Store iteration history
    weight_history = [slope]
    bias_history = [intercept]
    loss_history = []
    
    # Gradient descent iterations
    w, b = slope, intercept
    
    for i in range(iterations):
        # Predictions
        y_pred = w * x + b
        
        # Calculate loss (MSE)
        loss = np.mean((y - y_pred) ** 2)
        loss_history.append(loss)
        
        # Calculate gradients
        dw = -(2/n) * np.sum(x * (y - y_pred))
        db = -(2/n) * np.sum(y - y_pred)
        
        # Update parameters
        w = w - lr * dw
        b = b - lr * db
        
        if i < iterations - 1:  # Don't append final values twice
            weight_history.append(w)
            bias_history.append(b)
    
    # Final predictions
    y_pred_final = w * x + b
    y_pred_original = slope * x + intercept
    
    # Calculate comprehensive metrics
    residuals = y - y_pred_final
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - meanofy) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Additional metrics
    mae = np.mean(np.abs(residuals))
    mse = np.mean(residuals ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs(residuals / y)) * 100 if np.all(y != 0) else 0
    
    # Calculate confidence intervals (approximate)
    std_error = np.sqrt(mse / (n - 2))
    t_value = 2.262  # approximate t-value for 95% confidence
    margin_error = t_value * std_error
    
    return {
        'original_slope': slope,
        'original_intercept': intercept,
        'final_slope': w,
        'final_intercept': b,
        'y_pred_original': y_pred_original,
        'y_pred_final': y_pred_final,
        'weight_history': weight_history,
        'bias_history': bias_history,
        'loss_history': loss_history,
        'residuals': residuals,
        'r2': r2,
        'correlation': correlation,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape,
        'std_error': std_error,
        'margin_error': margin_error,
        'mean_x': meanofx,
        'mean_y': meanofy,
        'std_x': std_x,
        'std_y': std_y,
        'n_points': n
    }

def create_regression_plot(x, y, results, show_confidence=True):
    """Create an interactive regression plot with Plotly"""
    
    fig = go.Figure()
    
    # Original data points
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='markers',
        name='Original Data',
        marker=dict(size=10, color='blue', opacity=0.7),
        hovertemplate='X: %{x}<br>Y: %{y}<extra></extra>'
    ))
    
    # Original regression line (least squares)
    x_line = np.linspace(min(x), max(x), 100)
    y_line_original = results['original_slope'] * x_line + results['original_intercept']
    
    fig.add_trace(go.Scatter(
        x=x_line, y=y_line_original,
        mode='lines',
        name=f'Original: y = {results["original_slope"]:.3f}x + {results["original_intercept"]:.3f}',
        line=dict(color='red', width=2),
        hovertemplate='Original Regression<extra></extra>'
    ))
    
    # Updated regression line (after gradient descent)
    y_line_final = results['final_slope'] * x_line + results['final_intercept']
    
    fig.add_trace(go.Scatter(
        x=x_line, y=y_line_final,
        mode='lines',
        name=f'Updated: y = {results["final_slope"]:.3f}x + {results["final_intercept"]:.3f}',
        line=dict(color='green', width=2, dash='dash'),
        hovertemplate='Updated Regression<extra></extra>'
    ))
    
    # Predicted points
    fig.add_trace(go.Scatter(
        x=x, y=results['y_pred_final'],
        mode='markers',
        name='Predicted Values',
        marker=dict(size=8, color='orange', symbol='x'),
        hovertemplate='Predicted: %{y:.2f}<extra></extra>'
    ))
    
    # Confidence bands (approximate)
    if show_confidence:
        upper_bound = y_line_final + results['margin_error']
        lower_bound = y_line_final - results['margin_error']
        
        fig.add_trace(go.Scatter(
            x=np.concatenate([x_line, x_line[::-1]]),
            y=np.concatenate([upper_bound, lower_bound[::-1]]),
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% Confidence Interval',
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        title='Linear Regression Analysis',
        xaxis_title='X Values',
        yaxis_title='Y Values',
        hovermode='closest',
        template='plotly_white',
        height=500
    )
    
    return fig

def create_residuals_plot(x, results):
    """Create residuals analysis plots"""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Residuals vs Fitted', 'Residuals Distribution', 
                       'Q-Q Plot', 'Residuals vs X'),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    residuals = results['residuals']
    fitted = results['y_pred_final']
    
    # Residuals vs Fitted
    fig.add_trace(go.Scatter(
        x=fitted, y=residuals,
        mode='markers',
        name='Residuals',
        marker=dict(color='blue', opacity=0.6)
    ), row=1, col=1)
    
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
    
    # Residuals histogram
    fig.add_trace(go.Histogram(
        x=residuals,
        name='Distribution',
        marker_color='lightblue',
        opacity=0.7
    ), row=1, col=2)
    
    # Q-Q plot (simplified)
    from scipy import stats
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals)))
    sample_quantiles = np.sort(residuals)
    
    fig.add_trace(go.Scatter(
        x=theoretical_quantiles, y=sample_quantiles,
        mode='markers',
        name='Q-Q Plot',
        marker=dict(color='green', opacity=0.6)
    ), row=2, col=1)
    
    # Add reference line for Q-Q plot
    min_q, max_q = min(theoretical_quantiles), max(theoretical_quantiles)
    fig.add_trace(go.Scatter(
        x=[min_q, max_q], y=[min_q, max_q],
        mode='lines',
        name='Reference Line',
        line=dict(color='red', dash='dash')
    ), row=2, col=1)
    
    # Residuals vs X
    fig.add_trace(go.Scatter(
        x=x, y=residuals,
        mode='markers',
        name='Residuals vs X',
        marker=dict(color='purple', opacity=0.6)
    ), row=2, col=2)
    
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=2)
    
    fig.update_layout(
        height=600,
        showlegend=False,
        template='plotly_white'
    )
    
    return fig

def create_convergence_plot(results):
    """Create convergence analysis plot"""
    
    if len(results['loss_history']) <= 1:
        return None
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Loss Convergence', 'Weight Updates', 'Bias Updates')
    )
    
    iterations = list(range(len(results['loss_history'])))
    
    # Loss convergence
    fig.add_trace(go.Scatter(
        x=iterations, y=results['loss_history'],
        mode='lines+markers',
        name='Loss',
        line=dict(color='red', width=2)
    ), row=1, col=1)
    
    # Weight updates
    fig.add_trace(go.Scatter(
        x=list(range(len(results['weight_history']))), y=results['weight_history'],
        mode='lines+markers',
        name='Weight',
        line=dict(color='blue', width=2)
    ), row=1, col=2)
    
    # Bias updates
    fig.add_trace(go.Scatter(
        x=list(range(len(results['bias_history']))), y=results['bias_history'],
        mode='lines+markers',
        name='Bias',
        line=dict(color='green', width=2)
    ), row=1, col=3)
    
    fig.update_layout(
        height=400,
        showlegend=False,
        template='plotly_white'
    )
    
    return fig

def load_sample_datasets():
    """Load sample datasets for demonstration"""
    datasets = {
        "Default Data": {
            "x": [7, 9, 11, 15, 17, 21, 24],
            "y": [39, 56, 63, 73, 85, 93, 100],
            "description": "Simple linear relationship"
        },
        "House Prices": {
            "x": [1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400],
            "y": [150000, 180000, 210000, 240000, 270000, 300000, 330000, 360000],
            "description": "House size vs Price (sq ft vs $)"
        },
        "Temperature vs Sales": {
            "x": [10, 15, 20, 25, 30, 35, 40],
            "y": [20, 25, 35, 50, 70, 85, 95],
            "description": "Temperature vs Ice Cream Sales"
        },
        "Study Hours vs Grades": {
            "x": [1, 2, 3, 4, 5, 6, 7, 8],
            "y": [50, 55, 65, 70, 78, 85, 88, 92],
            "description": "Study hours vs Test scores"
        }
    }
    return datasets

def main():
    # Header
    st.markdown('<h1 class="main-header">📊  Linear Regression Calculator</h1>', unsafe_allow_html=True)
    st.markdown("### Comprehensive linear regression analysis with gradient descent optimization")
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Learning parameters
        st.subheader("Learning Parameters")
        lr = st.number_input("Learning Rate", 
                            min_value=0.0001, 
                            max_value=1.0, 
                            value=0.01, 
                            step=0.0001,
                            format="%.4f",
                            help="Controls how much to adjust weights during gradient descent")
        
        # Text field for gradient descent iterations with validation
        iterations_input = st.text_input(
            "Gradient Descent Iterations", 
            value="5",
            help="Enter a number between 1 and 10"
        )
        
        # Validate the input
        try:
            iterations = int(iterations_input)
            if iterations < 1 or iterations > 10:
                st.error("⚠️ Please enter a number between 1 and 10")
                iterations = 5  # Default fallback
            else:
                st.success(f"✅ Using {iterations} iterations")
        except ValueError:
            st.error("⚠️ Please enter a valid integer between 1 and 10")
            iterations = 5  # Default fallback
        
        # Data input methods
        st.subheader("📊 Data Input")
        input_method = st.radio("Choose input method:", 
                               ["Sample Datasets", "Manual Entry", "Upload CSV"])
        
        x, y = [], []
        
        if input_method == "Sample Datasets":
            datasets = load_sample_datasets()
            selected_dataset = st.selectbox("Select Dataset:", list(datasets.keys()))
            dataset = datasets[selected_dataset]
            x = dataset["x"]
            y = dataset["y"]
            st.info(f"📝 {dataset['description']}")
            st.write(f"**Data points:** {len(x)}")
        
        elif input_method == "Manual Entry":
            st.write("Enter comma-separated values:")
            x_input = st.text_area("X values:", value="7, 9, 11, 15, 17, 21, 24")
            y_input = st.text_area("Y values:", value="39, 56, 63, 73, 85, 93, 100")
            
            try:
                x = [float(val.strip()) for val in x_input.split(',')]
                y = [float(val.strip()) for val in y_input.split(',')]
                if len(x) != len(y):
                    st.error("⚠️ X and Y must have the same number of values")
                    return
            except ValueError:
                st.error("⚠️ Please enter valid numbers separated by commas")
                return
        
        else:  # Upload CSV
            uploaded_file = st.file_uploader("Choose CSV file", type="csv")
            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file)
                    if len(df.columns) >= 2:
                        x = df.iloc[:, 0].dropna().tolist()
                        y = df.iloc[:, 1].dropna().tolist()
                        st.success(f"✅ Loaded {len(x)} data points")
                        st.write("**Preview:**")
                        st.dataframe(df.head())
                    else:
                        st.error("⚠️ CSV must have at least 2 columns")
                        return
                except Exception as e:
                    st.error(f"⚠️ Error reading CSV: {str(e)}")
                    return
            else:
                st.info("📁 Please upload a CSV file")
                return
        
        # Visualization options
        st.subheader("🎨 Visualization Options")
        show_confidence = st.checkbox("Show Confidence Interval", value=True)
        show_residuals = st.checkbox("Show Residual Analysis", value=True)
        show_convergence = st.checkbox("Show Convergence Plot", value=iterations > 1)
    
    # Validate data
    if len(x) < 2:
        st.error("⚠️ Please provide at least 2 data points")
        return
    
    # Calculate regression
    with st.spinner("🔄 Calculating regression..."):
        results = calculate_regression(x, y, lr, iterations)
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Overview", "📊 Detailed Analysis", "🔍 Residual Analysis", "📉 Convergence", "📋 Data & Export"])
    
    with tab1:
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("R² Score", f"{results['r2']:.2f}", 
                     help="Coefficient of determination (0-1, higher is better)")
        
        with col2:
            st.metric("RMSE", f"{results['rmse']:.2f}",
                     help="Root Mean Square Error (lower is better)")
        
        with col3:
            st.metric("Correlation", f"{results['correlation']:.2f}",
                     help="Pearson correlation coefficient")
        
        with col4:
            st.metric("Data Points", f"{results['n_points']}",
                     help="Number of observations")
        
        st.markdown("---")
        
        # Main regression plot
        fig_main = create_regression_plot(x, y, results, show_confidence)
        st.plotly_chart(fig_main, use_container_width=True)
        
        # Summary statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📐 Regression Equations")
            st.markdown(f"""
            **Original (Least Squares):**  
            `y = {results['original_slope']:.2f}x + {results['original_intercept']:.2f}`
            
            **After Gradient Descent:**  
            `y = {results['final_slope']:.2f}x + {results['final_intercept']:.2f}`
            """)
        
        with col2:
            st.subheader("📏 Error Metrics")
            st.markdown(f"""
            - **MAE:** {results['mae']:.2f}
            - **MSE:** {results['mse']:.2f}
            - **RMSE:** {results['rmse']:.2f}
            - **MAPE:** {results['mape']:.2f}%
            """)
    
    with tab2:
        st.subheader("📊 Comprehensive Analysis")
        
        # Detailed statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 📈 Data Statistics")
            st.markdown(f"""
            - **Mean X:** {results['mean_x']:.2f}
            - **Mean Y:** {results['mean_y']:.2f}
            - **Std X:** {results['std_x']:.2f}
            - **Std Y:** {results['std_y']:.2f}
            - **Sample Size:** {results['n_points']}
            """)
        
        with col2:
            st.markdown("### 🎯 Model Performance")
            st.markdown(f"""
            - **R² Score:** {results['r2']:.2f}
            - **Correlation:** {results['correlation']:.2f}
            - **Std Error:** {results['std_error']:.2f}
            - **95% Margin:** ±{results['margin_error']:.2f}
            """)
        
        with col3:
            st.markdown("### ⚙️ Parameter Changes")
            slope_change = results['final_slope'] - results['original_slope']
            intercept_change = results['final_intercept'] - results['original_intercept']
            st.markdown(f"""
            - **Slope Change:** {slope_change:+.6f}
            - **Intercept Change:** {intercept_change:+.6f}
            - **Iterations:** {iterations}
            - **Learning Rate:** {lr}
            """)
        
        # Model interpretation
        st.markdown("### 🔍 Model Interpretation")
        
        if results['r2'] > 0.8:
            interpretation = "🟢 **Excellent fit** - The model explains most of the variance in the data."
        elif results['r2'] > 0.6:
            interpretation = "🟡 **Good fit** - The model explains a substantial portion of the variance."
        elif results['r2'] > 0.3:
            interpretation = "🟠 **Moderate fit** - The model has some explanatory power but significant variance remains."
        else:
            interpretation = "🔴 **Poor fit** - The linear model may not be appropriate for this data."
        
        st.markdown(interpretation)
        
        st.markdown(f"""
        **Slope Interpretation:** For every 1 unit increase in X, Y changes by approximately {results['final_slope']:.4f} units.
        
        **Intercept Interpretation:** When X = 0, the predicted value of Y is {results['final_intercept']:.4f}.
        """)
    
    with tab3:
        if show_residuals:
            st.subheader("🔍 Residual Analysis")
            st.markdown("Residual plots help assess model assumptions and identify potential issues.")
            
            fig_residuals = create_residuals_plot(x, results)
            st.plotly_chart(fig_residuals, use_container_width=True)
            
            # Residual statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 Residual Statistics")
                st.markdown(f"""
                - **Mean:** {np.mean(results['residuals']):.6f}
                - **Std Dev:** {np.std(results['residuals']):.6f}
                - **Min:** {np.min(results['residuals']):.4f}
                - **Max:** {np.max(results['residuals']):.4f}
                """)
            
            with col2:
                st.markdown("### ✅ Assumptions Check")
                residuals = results['residuals']
                
                # Check for patterns
                if abs(np.mean(residuals)) < 0.01:
                    st.success("✅ Residuals centered around zero")
                else:
                    st.warning("⚠️ Residuals not centered around zero")
                
                # Check for homoscedasticity (simplified)
                if np.std(residuals[:len(residuals)//2]) / np.std(residuals[len(residuals)//2:]) < 2:
                    st.success("✅ Relatively constant variance")
                else:
                    st.warning("⚠️ Possible heteroscedasticity")
        else:
            st.info("Enable 'Show Residual Analysis' in the sidebar to view detailed residual plots.")
    
    with tab4:
        if show_convergence and iterations > 1:
            st.subheader("📉 Convergence Analysis")
            st.markdown("Track how the model parameters and loss change during gradient descent optimization.")
            
            fig_convergence = create_convergence_plot(results)
            if fig_convergence:
                st.plotly_chart(fig_convergence, use_container_width=True)
                
                # Convergence statistics
                final_loss = results['loss_history'][-1]
                initial_loss = results['loss_history'][0]
                improvement = ((initial_loss - final_loss) / initial_loss) * 100
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Initial Loss", f"{initial_loss:.6f}")
                
                with col2:
                    st.metric("Final Loss", f"{final_loss:.6f}")
                
                with col3:
                    st.metric("Improvement", f"{improvement:.2f}%")
        else:
            if iterations == 1:
                st.info("Set iterations > 1 to view convergence analysis.")
            else:
                st.info("Enable 'Show Convergence Plot' in the sidebar to view optimization progress.")
    
    with tab5:
        st.subheader("📋 Data Table & Export")
        
        # Data comparison table
        df_results = pd.DataFrame({
            'X': x,
            'Y_Actual': y,
            'Y_Predicted': results['y_pred_final'],
            'Residual': results['residuals'],
            'Abs_Error': np.abs(results['residuals']),
            'Squared_Error': results['residuals'] ** 2
        })
        
        st.dataframe(df_results.round(4), use_container_width=True)
        
        # Export options
        col1, col2 = st.columns(2)
        
        with col1:
            # CSV export
            csv = df_results.to_csv(index=False)
            st.download_button(
                label="📥 Download Data as CSV",
                data=csv,
                file_name="regression_results.csv",
                mime="text/csv"
            )
        
        with col2:
            # Detailed report
            report = f"""
 Linear Regression Analysis Report
=========================================

Dataset Information:
- Number of observations: {results['n_points']}
- Mean X: {results['mean_x']:.6f}
- Mean Y: {results['mean_y']:.6f}
- Standard deviation X: {results['std_x']:.6f}
- Standard deviation Y: {results['std_y']:.6f}
- Correlation coefficient: {results['correlation']:.6f}

Model Parameters:
- Original slope (least squares): {results['original_slope']:.6f}
- Original intercept (least squares): {results['original_intercept']:.6f}
- Final slope (after GD): {results['final_slope']:.6f}
- Final intercept (after GD): {results['final_intercept']:.6f}

Optimization Settings:
- Learning rate: {lr}
- Iterations: {iterations}
- Parameter changes:
  - Slope change: {results['final_slope'] - results['original_slope']:+.6f}
  - Intercept change: {results['final_intercept'] - results['original_intercept']:+.6f}

Model Performance:
- R² Score: {results['r2']:.6f}
- Mean Absolute Error (MAE): {results['mae']:.6f}
- Mean Squared Error (MSE): {results['mse']:.6f}
- Root Mean Squared Error (RMSE): {results['rmse']:.6f}
- Mean Absolute Percentage Error (MAPE): {results['mape']:.2f}%
- Standard Error: {results['std_error']:.6f}
- 95% Confidence Margin: ±{results['margin_error']:.6f}

Final Equation: y = {results['final_slope']:.6f}x + {results['final_intercept']:.6f}

Residual Analysis:
- Mean residual: {np.mean(results['residuals']):.6f}
- Residual standard deviation: {np.std(results['residuals']):.6f}
- Min residual: {np.min(results['residuals']):.6f}
- Max residual: {np.max(results['residuals']):.6f}
"""
            
            st.download_button(
                label="📄 Download Detailed Report",
                data=report,
                file_name="regression_analysis_report.txt",
                mime="text/plain"
            )

if __name__ == "__main__":
    main()
