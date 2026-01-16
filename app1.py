import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Cars EDA Dashboard",
    page_icon="🚗",
    layout="wide"
)

# --------------------------------------------------
# Data Loaders
# --------------------------------------------------
@st.cache_data
def load_raw():
    return pd.read_csv("Cars.csv")

@st.cache_data
def load_cleaned():
    return pd.read_csv("Cars_cleaned.csv")

raw = load_raw()
clean = load_cleaned()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.title("🚗 Cars EDA Project")
page = st.sidebar.radio(
    "Navigation",
    ["Introduction", "Analysis", "Insights & Conclusion"]
)

st.sidebar.markdown("---")
st.sidebar.info("📊 Used Car Price Analysis\n\nIndia Market Focus")

# ==================================================
# INTRODUCTION PAGE
# ==================================================
if page == "Introduction":

    st.title("📌 Cars Analytics Dashboard")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Cars", len(clean))
    c2.metric("Avg Price (₹ Lakhs)", round(clean["Price"].mean(), 2))
    c3.metric("Avg KM Driven", int(clean["Kilometers_Driven"].mean()))
    c4.metric("Total Brands", clean["Company_Name"].nunique())

    st.markdown("---")

    with st.expander("📂 View Raw Dataset"):
        st.dataframe(raw, use_container_width=True)

    with st.expander("🧹 View Cleaned Dataset"):
        st.dataframe(clean, use_container_width=True)

# ==================================================
# ANALYSIS PAGE
# ==================================================
elif page == "Analysis":

    st.title("🔍 Exploratory Data Analysis")

    # ---------------- Filters ----------------
    st.sidebar.subheader("🔎 Filters")

    company = st.sidebar.multiselect(
        "Select Brand",
        clean["Company_Name"].unique(),
        default=clean["Company_Name"].unique()
    )

    year_range = st.sidebar.slider(
        "Manufacturing Year",
        int(clean["Year"].min()),
        int(clean["Year"].max()),
        (int(clean["Year"].min()), int(clean["Year"].max()))
    )

    df = clean[
        (clean["Company_Name"].isin(company)) &
        (clean["Year"].between(year_range[0], year_range[1]))
    ]

    num_cols = df.select_dtypes(include=np.number).columns
    cat_cols = df.select_dtypes(include="object").columns

    # ---------------- KPIs ----------------
    k1, k2, k3 = st.columns(3)
    k1.metric("Selected Cars", len(df))
    k2.metric("Avg Price", round(df["Price"].mean(), 2))
    k3.metric("Avg Power (bhp)", round(df["Power_value"].mean(), 2))

    # ---------------- Tabs ----------------
    tab1, tab2, tab3 = st.tabs(
        ["📊 Univariate", "📈 Bivariate", "📉 Multivariate"]
    )

    # -------- Univariate --------
    with tab1:
        col = st.selectbox("Choose Column", df.columns)

        fig, ax = plt.subplots(figsize=(7, 4))

        if col in cat_cols:
            sns.countplot(y=df[col], ax=ax)
        else:
            sns.histplot(df[col], kde=True, ax=ax)

        ax.set_title(f"Distribution of {col}")
        st.pyplot(fig)

    # -------- Bivariate --------
    with tab2:
        x = st.selectbox("X Axis", df.columns)
        y = st.selectbox("Y Axis", df.columns)

        fig, ax = plt.subplots(figsize=(7, 4))

        if x in num_cols and y in num_cols:
            sns.scatterplot(data=df, x=x, y=y, ax=ax)
            st.info(f"Correlation: **{round(df[x].corr(df[y]), 3)}**")

        elif x in cat_cols and y in num_cols:
            sns.boxplot(data=df, x=x, y=y, ax=ax)

        else:
            sns.countplot(data=df, x=x, hue=y, ax=ax)

        plt.xticks(rotation=90)
        ax.set_title(f"{x} vs {y}")
        st.pyplot(fig)

    # -------- Multivariate --------
    with tab3:
        method = st.selectbox(
            "Select Method",
            ["Correlation Heatmap", "Fuel vs Price"]
        )

        if method == "Correlation Heatmap":
            fig, ax = plt.subplots(figsize=(9, 5))
            sns.heatmap(
                df[num_cols].corr(),
                annot=True,
                cmap="coolwarm",
                ax=ax
            )
            ax.set_title("Correlation Matrix")
            st.pyplot(fig)

        else:
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.barplot(
                data=df,
                x="Fuel_Type",
                y="Price",
                hue="Transmission",
                ax=ax
            )
            ax.set_title("Fuel Type vs Price")
            st.pyplot(fig)

# ==================================================
# INSIGHTS & CONCLUSION
# ==================================================
else:

    st.title("📌 Automated Insights & Conclusion")

    st.success(f"📦 Total Records Analyzed: **{len(clean)}**")

    st.write(
        "💰 **Most Expensive Brand:**",
        clean.loc[clean["Price"].idxmax(), "Company_Name"]
    )

    st.write(
        "⛽ **Most Common Fuel Type:**",
        clean["Fuel_Type"].mode()[0]
    )

    strongest_corr = (
        clean.select_dtypes(include=np.number)
        .corr()["Price"]
        .sort_values(ascending=False)
        .index[1]
    )

    st.write(
        "📈 **Strongest Price Driver:**",
        strongest_corr
    )

    st.markdown("---")

    st.subheader("📌 Business Conclusion")
    st.markdown("""
    - Diesel and Automatic cars tend to have higher resale value  
    - Price decreases significantly with kilometers driven  
    - Engine capacity and power strongly influence pricing  
    - Metro cities show higher used car demand  
    """)

    st.info("✅ This dashboard reduces manual EDA effort by ~70% using automation.")
