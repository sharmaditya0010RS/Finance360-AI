import streamlit as st

THEME_CSS = """
<style>
:root {
    --f360-bg: #080F1C;
    --f360-panel: #111F33;
    --f360-panel-raised: #192B43;
    --f360-border: rgba(151, 174, 204, 0.19);
    --f360-text: #EEF4FD;
    --f360-muted: #A7B9D0;
    --f360-blue: #7398E8;
    --f360-green: #42C6A4;
    --f360-gold: #DDBB79;
    --f360-shadow: 0 18px 48px rgba(0, 0, 0, 0.20);
}

html {
    scroll-behavior: smooth;
}

[data-testid="stApp"],
[data-testid="stAppViewContainer"] {
    color: var(--f360-text);
    background:
        radial-gradient(
            ellipse 65% 48% at 88% 0%,
            rgba(92, 119, 207, 0.20),
            transparent 74%
        ),
        radial-gradient(
            ellipse 46% 46% at 3% 55%,
            rgba(34, 142, 136, 0.09),
            transparent 76%
        ),
        radial-gradient(
            ellipse 50% 40% at 85% 100%,
            rgba(77, 79, 153, 0.09),
            transparent 75%
        ),
        linear-gradient(
            145deg,
            #080F1C 0%,
            #0B1526 48%,
            #0A1323 100%
        );
}

[data-testid="stHeader"] {
    background: rgba(8, 15, 28, 0.82);
    backdrop-filter: blur(18px);
    border-bottom: 1px solid rgba(151, 174, 204, 0.08);
}

[data-testid="stMainBlockContainer"] {
    max-width: 1600px;
    padding: 2.3rem clamp(1.2rem, 3vw, 3.1rem) 4rem;
}

[data-testid="stSidebar"] {
    background:
        radial-gradient(
            ellipse at 0% 0%,
            rgba(115, 152, 232, 0.13),
            transparent 55%
        ),
        linear-gradient(
            180deg,
            #101F34 0%,
            #0D192B 55%,
            #0A1423 100%
        );
    border-right: 1px solid var(--f360-border);
}

[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    padding-top: 1.2rem;
}

[data-testid="stSidebar"] h1 {
    color: #F5F8FF;
    font-size: 1.23rem;
    font-weight: 800;
    letter-spacing: 0.08em;
}

[data-testid="stSidebarNav"] {
    padding-top: 0.7rem;
}

[data-testid="stSidebarNav"] a {
    display: flex;
    align-items: center;
    min-height: 46px;
    margin: 0.2rem 0.45rem;
    padding: 0.65rem 0.85rem;
    border-radius: 11px;
    border: 1px solid transparent;
    color: #A7B9D0;
    font-weight: 560;
    transition:
        background 180ms ease,
        border-color 180ms ease,
        color 180ms ease;
}

[data-testid="stSidebarNav"] a:hover {
    color: #F4F8FF;
    background: rgba(115, 152, 232, 0.12);
    border-color: rgba(115, 152, 232, 0.18);
}

[data-testid="stSidebarNav"] a[aria-current="page"] {
    color: #F7FAFF;
    background: linear-gradient(
        100deg,
        rgba(115, 152, 232, 0.24),
        rgba(66, 198, 164, 0.09)
    );
    border-color: rgba(115, 152, 232, 0.38);
    box-shadow:
        inset 3px 0 0 #8FAAF0,
        0 7px 18px rgba(0, 0, 0, 0.10);
}

[data-testid="stMain"] h1 {
    color: #F7FAFF;
    font-size: clamp(2rem, 3vw, 2.7rem);
    font-weight: 780;
    letter-spacing: -0.04em;
    line-height: 1.2;
}

[data-testid="stMain"] h2 {
    color: #EAF1FB;
    font-weight: 720;
    letter-spacing: -0.025em;
}

[data-testid="stMain"] h3 {
    color: #DFEAF8;
    font-weight: 680;
}

[data-testid="stCaptionContainer"] {
    color: var(--f360-muted);
    line-height: 1.55;
}

[data-testid="stMetric"] {
    position: relative;
    isolation: isolate;
    min-height: 128px;
    padding: 1.3rem 1.35rem;
    border: 1px solid var(--f360-border);
    border-radius: 16px;
    background: linear-gradient(
        145deg,
        rgba(25, 44, 68, 0.97),
        rgba(15, 29, 48, 0.97)
    );
    box-shadow: var(--f360-shadow);
    transition:
        transform 180ms ease,
        border-color 180ms ease,
        box-shadow 180ms ease;
}

[data-testid="stMetric"]::before {
    content: "";
    position: absolute;
    top: -1px;
    left: 0;
    right: 0;
    height: 3px;
    border-radius: 16px 16px 0 0;
    background: linear-gradient(
        90deg,
        #7398E8 0%,
        #8CA7EC 38%,
        #42C6A4 76%,
        #DDBB79 100%
    );
    z-index: 2;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    border-color: rgba(115, 152, 232, 0.45);
    box-shadow: 0 23px 52px rgba(0, 0, 0, 0.25);
}

[data-testid="stMetricLabel"] {
    color: #A7B9D0;
    font-size: 0.84rem;
    font-weight: 570;
}

[data-testid="stMetricValue"] {
    color: #F8FAFF;
    font-size: clamp(1.4rem, 2vw, 2rem);
    font-weight: 760;
    letter-spacing: -0.035em;
    overflow-wrap: anywhere;
}

[data-testid="stMetricDelta"] {
    font-weight: 650;
}

[data-testid="stTabs"] [data-baseweb="tab-list"] {
    display: flex;
    gap: 0.6rem;
    padding: 0.55rem;
    border: 1px solid rgba(151, 174, 204, 0.22);
    border-radius: 15px;
    background: rgba(16, 30, 49, 0.94);
    overflow-x: auto;
    scrollbar-width: none;
}

[data-testid="stTabs"] [data-baseweb="tab-list"]::-webkit-scrollbar {
    display: none;
}

[data-testid="stTabs"] [data-baseweb="tab"] {
    flex: 0 0 auto;
    min-height: 43px;
    padding: 0.72rem 1.1rem;
    border: 1px solid rgba(151, 174, 204, 0.20);
    border-radius: 10px;
    color: #B6C6DA;
    background: rgba(25, 43, 67, 0.72);
    font-size: 0.9rem;
    font-weight: 650;
    white-space: nowrap;
    transition:
        color 160ms ease,
        background 160ms ease,
        border-color 160ms ease,
        box-shadow 160ms ease;
}

[data-testid="stTabs"] [data-baseweb="tab"]:hover {
    color: #F8FAFF;
    background: rgba(115, 152, 232, 0.15);
    border-color: rgba(115, 152, 232, 0.40);
}

[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {
    color: #F8FAFF;
    background: linear-gradient(
        110deg,
        rgba(115, 152, 232, 0.35),
        rgba(66, 198, 164, 0.16)
    );
    border-color: rgba(143, 170, 240, 0.72);
    box-shadow:
        0 5px 18px rgba(0, 0, 0, 0.17),
        inset 0 0 0 1px rgba(143, 170, 240, 0.15);
}

[data-testid="stTabs"] [data-baseweb="tab"]:focus-visible {
    outline: 2px solid #A7BDF3;
    outline-offset: 2px;
}

[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
    background: transparent;
}

[data-testid="stTabs"] [data-baseweb="tab-panel"] {
    padding-top: 1.4rem;
}

[data-testid="stPlotlyChart"] {
    min-width: 0;
    max-width: 100%;
    padding: 0.4rem;
    border: 1px solid var(--f360-border);
    border-radius: 16px;
    background: linear-gradient(
        145deg,
        #14243A,
        #101C2D
    );
    box-shadow: 0 12px 34px rgba(0, 0, 0, 0.14);
}

[data-testid="stPlotlyChart"] .js-plotly-plot,
[data-testid="stPlotlyChart"] .plot-container {
    max-width: 100%;
}

[data-testid="stPlotlyChart"] .modebar {
    background: transparent !important;
}

[data-testid="stPlotlyChart"] .modebar-btn {
    border-radius: 6px;
}

[data-testid="stDataFrame"],
[data-testid="stTable"] {
    border: 1px solid var(--f360-border);
    border-radius: 12px;
    background: var(--f360-panel);
}

[data-testid="stForm"] {
    padding: 1.4rem;
    border: 1px solid var(--f360-border);
    border-radius: 16px;
    background: linear-gradient(
        150deg,
        rgba(25, 43, 67, 0.97),
        rgba(14, 28, 46, 0.97)
    );
    box-shadow: var(--f360-shadow);
}

.stButton > button,
.stDownloadButton > button,
[data-testid="stFormSubmitButton"] > button {
    min-height: 45px;
    border-radius: 11px;
    font-weight: 680;
    transition:
        background 160ms ease,
        border-color 160ms ease,
        box-shadow 160ms ease,
        transform 160ms ease;
}

.stButton > button[kind="primary"],
.stDownloadButton > button[kind="primary"],
[data-testid="stFormSubmitButton"] > button[kind="primary"] {
    color: #0A1627;
    border: 1px solid #A7BDF3;
    background: linear-gradient(
        110deg,
        #91ACEE,
        #70CFB9
    );
    box-shadow: 0 9px 24px rgba(115, 152, 232, 0.16);
}

.stButton > button:not([kind="primary"]),
.stDownloadButton > button:not([kind="primary"]) {
    color: #E3ECFA;
    background: #1A2D46;
    border: 1px solid rgba(151, 174, 204, 0.30);
}

.stButton > button:hover,
.stDownloadButton > button:hover,
[data-testid="stFormSubmitButton"] > button:hover {
    transform: translateY(-1px);
    border-color: #91ACEE;
    box-shadow: 0 10px 26px rgba(115, 152, 232, 0.17);
}

.stButton > button:focus-visible,
.stDownloadButton > button:focus-visible,
[data-testid="stFormSubmitButton"] > button:focus-visible,
[data-testid="stSidebarNav"] a:focus-visible {
    outline: 2px solid #A7BDF3;
    outline-offset: 3px;
}

[data-baseweb="input"] > div,
[data-baseweb="select"] > div,
[data-baseweb="textarea"] > div {
    border-radius: 10px;
    background: #17283F;
    border-color: rgba(151, 174, 204, 0.28);
}

[data-baseweb="input"]:focus-within,
[data-baseweb="select"]:focus-within,
[data-baseweb="textarea"]:focus-within {
    box-shadow: 0 0 0 2px rgba(115, 152, 232, 0.22);
}

[data-baseweb="tag"] {
    border-radius: 7px;
    background: rgba(115, 152, 232, 0.20);
    border: 1px solid rgba(115, 152, 232, 0.32);
}

[data-baseweb="tag"] span {
    color: #E8F0FF;
}

[data-testid="stExpander"] {
    border: 1px solid var(--f360-border);
    border-radius: 12px;
    background: rgba(20, 36, 58, 0.85);
}

[data-testid="stAlert"] {
    border-radius: 12px;
}

[data-testid="stCodeBlock"] {
    border: 1px solid var(--f360-border);
    border-radius: 12px;
}

a {
    color: #A7BDF3;
}

a:hover {
    color: #79D8BF;
}

@media (max-width: 900px) {
    [data-testid="stMainBlockContainer"] {
        padding: 1.5rem 1rem 3rem;
    }

    [data-testid="stMetric"] {
        min-height: 115px;
        padding: 1rem;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.45rem;
    }

    [data-testid="stTabs"] [data-baseweb="tab"] {
        padding: 0.65rem 0.8rem;
    }

    [data-testid="stPlotlyChart"] {
        padding: 0.2rem;
    }
}

@media (prefers-reduced-motion: reduce) {
    *,
    *::before,
    *::after {
        transition-duration: 0.01ms !important;
        animation-duration: 0.01ms !important;
        scroll-behavior: auto !important;
    }
}

[data-testid="stHorizontalBlock"],
[data-testid="stColumn"],
[data-testid="stVerticalBlock"] {
    min-width: 0;
}

[data-testid="stPlotlyChart"] {
    width: 100%;
    max-width: 100%;
    min-width: 0;
    box-sizing: border-box;
    padding: 0;
    overflow: clip;
}

[data-testid="stPlotlyChart"] > div {
    min-width: 0;
    max-width: 100%;
}

[data-testid="stPlotlyChart"] .js-plotly-plot,
[data-testid="stPlotlyChart"] .plot-container {
    max-width: 100%;
}

@media (max-width: 1100px) {
    [data-testid="stMainBlockContainer"] {
        padding-left: 1rem;
        padding-right: 1rem;
    }
}

</style>
"""


def apply_theme() -> None:
    st.markdown(THEME_CSS, unsafe_allow_html=True)