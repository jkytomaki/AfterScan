"""Design tokens and QSS stylesheet for the modernized AfterScan UI.

Tokens mirror `design/modern-ui-handoff/project/app.css`. Where the design uses
`oklch()` we substitute the closest sRGB hex equivalent — Qt doesn't speak
oklch and the visual difference is imperceptible at these chromas.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtGui import QPalette, QColor
from PySide6.QtWidgets import QApplication


@dataclass(frozen=True)
class Tokens:
    bg_app: str
    bg_panel: str
    bg_input: str
    bg_hover: str
    bg_active: str
    fg_1: str
    fg_2: str
    fg_3: str
    fg_4: str
    line_1: str
    line_2: str
    line_3: str
    accent: str
    accent_fg: str
    good: str


DARK = Tokens(
    bg_app="#1a1816",
    bg_panel="#221f1c",
    bg_input="#14120f",
    bg_hover="rgba(255,255,255,0.04)",
    bg_active="rgba(255,255,255,0.08)",
    fg_1="#ece8e1",
    fg_2="rgba(236,232,225,0.72)",
    fg_3="rgba(236,232,225,0.50)",
    fg_4="rgba(236,232,225,0.30)",
    line_1="rgba(255,255,255,0.06)",
    line_2="rgba(255,255,255,0.10)",
    line_3="rgba(255,255,255,0.16)",
    accent="#e8a14a",
    accent_fg="#1a1208",
    good="#5fb985",
)


def stylesheet(t: Tokens = DARK) -> str:
    return f"""
    * {{ outline: 0; }}

    QWidget {{
        background: {t.bg_app};
        color: {t.fg_1};
        font-family: "Inter", "Segoe UI", "DejaVu Sans", sans-serif;
        font-size: 12px;
    }}

    QMainWindow, #stage {{ background: {t.bg_app}; }}

    /* Top bar */
    #topbar {{
        background: {t.bg_app};
        border-bottom: 1px solid {t.line_1};
    }}

    #brand {{ font-weight: 600; font-size: 14px; }}
    #brand-small {{ color: {t.fg_3}; font-weight: 400; }}
    #brand-mark {{
        background: {t.accent};
        color: {t.accent_fg};
        font-weight: 800;
        font-size: 11px;
        border-radius: 6px;
    }}

    QLabel#crumb-lbl {{
        color: {t.fg_3};
        font-size: 11px;
    }}
    QLabel#crumb-path {{
        color: {t.fg_1};
        font-family: "JetBrains Mono", "DejaVu Sans Mono", monospace;
        font-size: 11px;
    }}
    QLabel#crumb-path[dim="true"] {{ color: {t.fg_3}; }}

    QFrame#sep {{ background: {t.line_2}; max-width: 1px; }}

    /* Status pill */
    QLabel#status-pill {{
        background: {t.bg_input};
        border: 1px solid {t.line_1};
        border-radius: 14px;
        padding: 4px 12px;
        color: {t.fg_2};
        font-size: 12px;
    }}

    /* Buttons */
    QPushButton {{
        background: {t.bg_input};
        color: {t.fg_1};
        border: 1px solid {t.line_2};
        border-radius: 6px;
        padding: 6px 12px;
        font-size: 12px;
        font-weight: 500;
    }}
    QPushButton:hover {{ background: {t.bg_hover}; border-color: {t.line_3}; }}
    QPushButton:pressed {{ background: {t.bg_active}; }}
    QPushButton:disabled {{ color: {t.fg_4}; }}

    QPushButton[variant="primary"] {{
        background: {t.accent};
        color: {t.accent_fg};
        border: 1px solid transparent;
        font-weight: 600;
    }}
    QPushButton[variant="primary"]:hover {{ background: #f1b162; }}
    QPushButton[variant="primary"]:pressed {{ background: #d49141; }}

    QPushButton[variant="ghost"] {{
        background: transparent;
        border: 1px solid transparent;
        color: {t.fg_2};
    }}
    QPushButton[variant="ghost"]:hover {{ background: {t.bg_hover}; color: {t.fg_1}; }}

    QPushButton[size="lg"] {{
        padding: 8px 16px;
        font-size: 13px;
    }}

    /* Workflow steps */
    QFrame#steps {{
        background: {t.bg_input};
        border: 1px solid {t.line_1};
        border-radius: 8px;
    }}
    QPushButton#step {{
        background: transparent;
        border: 1px solid transparent;
        color: {t.fg_3};
        padding: 6px 12px;
        border-radius: 5px;
        text-align: left;
    }}
    QPushButton#step:hover {{ color: {t.fg_1}; }}
    QPushButton#step[on="true"] {{
        background: {t.bg_app};
        color: {t.fg_1};
    }}

    QLabel#step-num {{
        background: {t.bg_active};
        color: {t.fg_2};
        font-family: "JetBrains Mono", monospace;
        font-size: 10px;
        border-radius: 4px;
        min-width: 16px;
        max-width: 16px;
        min-height: 16px;
        max-height: 16px;
        qproperty-alignment: AlignCenter;
    }}
    QLabel#step-num[on="true"] {{ background: {t.accent}; color: {t.accent_fg}; }}
    QLabel#step-num[done="true"] {{ background: {t.good}; color: {t.accent_fg}; }}

    /* Inspector */
    QFrame#inspector {{
        background: {t.bg_panel};
        border-left: 1px solid {t.line_1};
    }}
    QFrame#insp-tabs {{ border-bottom: 1px solid {t.line_1}; }}
    QPushButton#insp-tab {{
        background: transparent;
        border: 1px solid transparent;
        color: {t.fg_3};
        font-size: 11px;
        font-weight: 500;
        padding: 6px;
    }}
    QPushButton#insp-tab[on="true"] {{
        background: {t.bg_active};
        color: {t.fg_1};
        border-radius: 5px;
    }}

    QLabel.section-hd, QLabel#section-hd {{
        color: {t.fg_3};
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 1px;
    }}

    /* Inputs */
    QLineEdit, QComboBox {{
        background: {t.bg_input};
        border: 1px solid {t.line_2};
        border-radius: 4px;
        padding: 4px 8px;
        color: {t.fg_1};
        selection-background-color: {t.accent};
        selection-color: {t.accent_fg};
    }}
    QLineEdit:focus, QComboBox:focus {{ border-color: {t.accent}; }}
    QLineEdit:read-only {{ color: {t.fg_2}; }}

    /* Sliders */
    QSlider::groove:horizontal {{
        background: {t.bg_input};
        border: 1px solid {t.line_1};
        height: 4px;
        border-radius: 2px;
    }}
    QSlider::handle:horizontal {{
        background: {t.fg_1};
        border: 2px solid {t.bg_panel};
        width: 12px;
        height: 12px;
        margin: -6px 0;
        border-radius: 8px;
    }}

    /* Scrollbars */
    QScrollBar:vertical {{
        background: transparent;
        width: 8px;
        margin: 0;
    }}
    QScrollBar::handle:vertical {{
        background: {t.line_2};
        border-radius: 4px;
        min-height: 24px;
    }}
    QScrollBar::handle:vertical:hover {{ background: {t.line_3}; }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}

    QScrollBar:horizontal {{ background: transparent; height: 8px; }}
    QScrollBar::handle:horizontal {{ background: {t.line_2}; border-radius: 4px; min-width: 24px; }}
    QScrollBar::handle:horizontal:hover {{ background: {t.line_3}; }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}

    /* Preview placeholder */
    QFrame#preview-wrap {{ background: {t.bg_app}; }}
    QFrame#preview {{
        background: #000;
        border: 1px solid {t.line_2};
        border-radius: 6px;
    }}

    /* Filmstrip / queue */
    QFrame#timeline {{ background: {t.bg_app}; border-top: 1px solid {t.line_1}; }}
    QFrame#queue {{ background: {t.bg_panel}; border-top: 1px solid {t.line_1}; }}
    QFrame#queue-hd {{ border-bottom: 1px solid {t.line_1}; }}
    """


def apply_theme(app: QApplication, tokens: Tokens = DARK) -> None:
    app.setStyleSheet(stylesheet(tokens))
    palette = app.palette()
    palette.setColor(QPalette.Window, QColor(tokens.bg_app))
    palette.setColor(QPalette.WindowText, QColor(tokens.fg_1))
    palette.setColor(QPalette.Base, QColor(tokens.bg_input))
    palette.setColor(QPalette.Text, QColor(tokens.fg_1))
    palette.setColor(QPalette.Highlight, QColor(tokens.accent))
    palette.setColor(QPalette.HighlightedText, QColor(tokens.accent_fg))
    app.setPalette(palette)
