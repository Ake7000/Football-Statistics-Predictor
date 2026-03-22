"""
ui_app.py

Desktop UI for the football prediction project.

- Football pitch background.
- Two team search bars (Home / Away) with fuzzy suggestions from team_index.py.
- Suggestions shown as a popup under the search bar (QCompleter), search bar
  stays fixed.
- User can only trigger Battle if both teams are picked from suggestions
  (backend will receive team_id).
- On Battle:
    - run the backend pipeline (build_single_input_row -> aggregated -> predict)
      in a background thread
    - show a small loading dialog "Predicting the future..."
    - read predictions/*/report.txt
    - display the middle 'medie' value for each target in the stats cards
"""

import sys
from pathlib import Path
from typing import Optional, List, Dict

from PyQt5.QtCore import Qt, pyqtSignal, QStringListModel, QThread, QObject
from PyQt5.QtGui import QPainter, QPixmap, QFont
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QFrame,
    QMessageBox,
    QSizePolicy,
    QCompleter,
    QDialog,
    QProgressBar,
)

from team_index import ensure_team_index, search_teams, TeamEntry
from backend_bridge import run_prediction_pipeline, BackendError

# ---------- Paths & background image helper ----------

BASE_DIR = Path(__file__).resolve().parent


def find_background_image() -> Optional[Path]:
    """
    Try to find a suitable background image in the project folder.
    """
    candidates = [
        "pitch_bg.png",
        "pitch_bg.jpg",
        "football_pitch.png",
        "football_pitch.jpg",
    ]
    for name in candidates:
        p = BASE_DIR / name
        if p.is_file():
            return p
    return None


# ---------- Custom central widget with pitch background ----------

class PitchBackgroundWidget(QWidget):
    """
    Central widget that paints a football pitch image as background and hosts
    the actual UI layout on top.
    """

    def __init__(self, bg_path: Optional[Path], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._bg_pixmap: Optional[QPixmap] = None
        if bg_path is not None and bg_path.is_file():
            self._bg_pixmap = QPixmap(str(bg_path))

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(20, 20, 20, 20)
        self.main_layout.setSpacing(15)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._bg_pixmap is None or self._bg_pixmap.isNull():
            return

        painter = QPainter(self)
        scaled = self._bg_pixmap.scaled(
            self.size(),
            Qt.KeepAspectRatioByExpanding,
            Qt.SmoothTransformation,
        )
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        painter.drawPixmap(x, y, scaled)


# ---------- Team search widget (with QCompleter) ----------

class TeamSearchWidget(QWidget):
    """
    Reusable widget:
    - One QLineEdit with placeholder text.
    - QCompleter popup with fuzzy suggestions (no permanent list in layout).
    - Stores selected team_id and team_name only when user chooses from
      suggestions (activated signal from completer).
    """

    selectionChanged = pyqtSignal(object)  # emits selected_team_id or None

    def __init__(
        self,
        placeholder: str,
        teams: List[TeamEntry],
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._teams = teams
        self._selected_team_id: Optional[int] = None
        self._selected_team_name: Optional[str] = None

        self._current_name_to_id: Dict[str, int] = {}  # mapping name -> id for current suggestions

        self._build_ui(placeholder)
        self._connect_signals()

    # --- Public properties ---

    @property
    def selected_team_id(self) -> Optional[int]:
        return self._selected_team_id

    @property
    def selected_team_name(self) -> Optional[str]:
        return self._selected_team_name

    # --- UI setup ---

    def _build_ui(self, placeholder: str) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.line_edit = QLineEdit()
        self.line_edit.setPlaceholderText(placeholder)
        layout.addWidget(self.line_edit)

        # QCompleter popup for suggestions
        self._model = QStringListModel([], self)
        self.completer = QCompleter(self._model, self)
        self.completer.setCaseSensitivity(Qt.CaseInsensitive)
        self.completer.setFilterMode(Qt.MatchContains)
        self.completer.setCompletionMode(QCompleter.PopupCompletion)
        self.line_edit.setCompleter(self.completer)

    def _connect_signals(self) -> None:
        # textEdited = only when user manually edits text (not when setText() is called)
        self.line_edit.textEdited.connect(self._on_text_edited)
        # when user chooses an option from the popup
        self.completer.activated[str].connect(self._on_completer_activated)

    # --- Internal logic ---

    def _reset_selection(self) -> None:
        """Clear the selected team when user edits text manually."""
        if self._selected_team_id is not None:
            self._selected_team_id = None
            self._selected_team_name = None
            self.selectionChanged.emit(None)

    def _on_text_edited(self, text: str) -> None:
        # any manual edit invalidates previous selection
        self._reset_selection()

        text = text.strip()
        if not text:
            self._model.setStringList([])
            self._current_name_to_id.clear()
            return

        matches = search_teams(text, self._teams, limit=10)

        names = [t.name for t in matches]
        self._current_name_to_id = {t.name: t.team_id for t in matches}
        self._model.setStringList(names)

    def _on_completer_activated(self, text: str) -> None:
        """User selected a team from suggestions (single click)."""
        team_id = self._current_name_to_id.get(text)
        if team_id is None:
            return

        # setText here does NOT trigger textEdited, so selection is preserved
        self.line_edit.setText(text)

        self._selected_team_id = int(team_id)
        self._selected_team_name = str(text)

        self.selectionChanged.emit(self._selected_team_id)


# ---------- Worker pentru pipeline (rulează în QThread) ----------

class PredictionWorker(QObject):
    finished = pyqtSignal(dict)       # summary dict
    error = pyqtSignal(str)          # error message

    def __init__(self, home_id: int, away_id: int, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.home_id = home_id
        self.away_id = away_id

    def run(self):
        try:
            summary = run_prediction_pipeline(self.home_id, self.away_id)
        except BackendError as e:
            self.error.emit(str(e))
        except Exception as e:
            self.error.emit(f"Unexpected error: {e}")
        else:
            self.finished.emit(summary)


# ---------- Main window ----------

class MainWindow(QMainWindow):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        # Load teams index (builds it if missing)
        self.teams = ensure_team_index()

        # labels inside stats frames (we keep a reference to update text later)
        self.home_stats_label: Optional[QLabel] = None
        self.away_stats_label: Optional[QLabel] = None

        # worker & loading dialog
        self._worker_thread: Optional[QThread] = None
        self._worker: Optional[PredictionWorker] = None
        self._loading_dialog: Optional[QDialog] = None

        self._build_ui()

    def _build_ui(self) -> None:
        self.setWindowTitle("Football Match Predictor")
        self.resize(1200, 800)

        bg_path = find_background_image()
        self.pitch_widget = PitchBackgroundWidget(bg_path, parent=self)
        self.setCentralWidget(self.pitch_widget)

        layout = self.pitch_widget.main_layout

        # Title
        title_label = QLabel("UEFA Match Battle")
        title_label.setStyleSheet("color: white; font-size: 24px; font-weight: bold;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # Row: [Home search] [Battle] [Away search] on the same line
        top_row = QHBoxLayout()
        top_row.setSpacing(20)

        self.home_search = TeamSearchWidget("Type home team name", self.teams)
        self._style_search_widget(self.home_search)
        top_row.addWidget(self.home_search, stretch=3)

        self.battle_btn = QPushButton("Battle")
        self.battle_btn.setEnabled(False)
        self.battle_btn.setFixedHeight(50)
        self.battle_btn.setStyleSheet(
            "QPushButton {"
            "  background-color: #ffcc00;"
            "  color: #003300;"
            "  font-weight: bold;"
            "  font-size: 16px;"
            "  border-radius: 8px;"
            "  padding: 8px 16px;"
            "}"
            "QPushButton:disabled {"
            "  background-color: #888888;"
            "  color: #444444;"
            "}"
        )
        top_row.addWidget(self.battle_btn, stretch=1, alignment=Qt.AlignCenter)

        self.away_search = TeamSearchWidget("Type away team name", self.teams)
        self._style_search_widget(self.away_search)
        top_row.addWidget(self.away_search, stretch=3)

        layout.addLayout(top_row)

        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setStyleSheet("color: rgba(255, 255, 255, 160);")
        layout.addWidget(separator)

        # Some free space before stats cards
        layout.addStretch(1)

        # Stats area: Home (left) / Away (right)
        stats_row = QHBoxLayout()
        stats_row.setSpacing(40)

        self.home_stats_frame = self._create_stats_frame("Home statistics", side="HOME")
        self.away_stats_frame = self._create_stats_frame("Away statistics", side="AWAY")

        stats_row.addWidget(self.home_stats_frame)
        stats_row.addWidget(self.away_stats_frame)

        layout.addLayout(stats_row)

        # Free space below stats cards, so they are vertically centered-ish
        layout.addStretch(1)

        # Connect signals
        self.home_search.selectionChanged.connect(self._on_selection_changed)
        self.away_search.selectionChanged.connect(self._on_selection_changed)
        self.battle_btn.clicked.connect(self._on_battle_clicked)

    def _style_search_widget(self, widget: TeamSearchWidget) -> None:
        widget.line_edit.setStyleSheet(
            "QLineEdit {"
            "  padding: 6px;"
            "  border-radius: 6px;"
            "  border: 1px solid #cccccc;"
            "  font-size: 13px;"
            "}"
        )

    def _create_stats_frame(self, title: str, side: str) -> QFrame:
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setStyleSheet(
            "QFrame {"
            "  background-color: rgba(0, 0, 0, 160);"
            "  border-radius: 12px;"
            "}"
        )
        frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        frame.setMinimumHeight(140)

        layout = QVBoxLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        label_title = QLabel(title)
        label_title.setStyleSheet("color: white; font-size: 16px; font-weight: bold;")
        layout.addWidget(label_title)

        placeholder = QLabel("No data yet.\nPress Battle to generate predictions.")
        placeholder.setStyleSheet("color: white; font-size: 16px;")
        placeholder.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        placeholder.setWordWrap(True)
        placeholder.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        layout.addWidget(placeholder, stretch=1)

        if side.upper() == "HOME":
            self.home_stats_label = placeholder
        else:
            self.away_stats_label = placeholder

        return frame

    # ---------- Logic: selection & Battle ----------

    def _on_selection_changed(self, _):
        """
        Called whenever either home or away selection changes.
        Enables/disables the Battle button accordingly.
        """
        home_id = self.home_search.selected_team_id
        away_id = self.away_search.selected_team_id

        enable = (
            home_id is not None
            and away_id is not None
            and home_id != away_id
        )
        self.battle_btn.setEnabled(enable)

    def _format_side_stats(self, side: str, metrics: Dict[str, float]) -> str:
        """
        Convert metrics dict into a multi-line string for one side.
        metrics keys: GOALS, CORNERS, YELLOWCARDS, SHOTS_ON_TARGET, FOULS, OFFSIDES, REDCARDS
        """
        order = [
            "GOALS",
            "CORNERS",
            "YELLOWCARDS",
            "SHOTS_ON_TARGET",
            "FOULS",
            "OFFSIDES",
            "REDCARDS",
        ]
        nice_names = {
            "GOALS": "Goals",
            "CORNERS": "Corners",
            "YELLOWCARDS": "Yellow cards",
            "SHOTS_ON_TARGET": "Shots on target",
            "FOULS": "Fouls",
            "OFFSIDES": "Offsides",
            "REDCARDS": "Red cards",
        }

        lines = []
        for key in order:
            display = nice_names.get(key, key.title())
            val = metrics.get(key)
            if val is None:
                txt = "n/a"
            else:
                txt = f"{val:.2f}"
            lines.append(f"{display}: {txt}")

        return "\n".join(lines)

    def _update_stats_ui(self, summary: Dict[str, Dict[str, float]]) -> None:
        """
        Update the two stats cards using the summary from backend_bridge.run_prediction_pipeline.
        """
        home_metrics = summary.get("HOME", {})
        away_metrics = summary.get("AWAY", {})

        if self.home_stats_label is not None:
            self.home_stats_label.setText(self._format_side_stats("HOME", home_metrics))

        if self.away_stats_label is not None:
            self.away_stats_label.setText(self._format_side_stats("AWAY", away_metrics))

    # ---------- Loading dialog helpers ----------

    def _show_loading_dialog(self):
        if self._loading_dialog is not None:
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Please wait")
        dlg.setWindowModality(Qt.ApplicationModal)
        dlg.setWindowFlags(
            dlg.windowFlags()
            & ~Qt.WindowContextHelpButtonHint
            & ~Qt.WindowMaximizeButtonHint
        )

        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        label = QLabel("Predicting the future...")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(label)

        bar = QProgressBar()
        bar.setRange(0, 0)  # infinite / busy mode
        bar.setTextVisible(False)
        layout.addWidget(bar)

        dlg.setFixedSize(300, 120)
        self._loading_dialog = dlg
        dlg.show()

    def _hide_loading_dialog(self):
        if self._loading_dialog is not None:
            self._loading_dialog.close()
            self._loading_dialog = None

    def _clear_worker_refs(self):
        self._worker_thread = None
        self._worker = None

    # ---------- Battle click & worker callbacks ----------

    def _on_battle_clicked(self):
        """
        On Battle:
        - run backend pipeline for (home_team_id, away_team_id) in background
        - show loading dialog while it runs
        - update stats cards with predictions
        """
        home_id = self.home_search.selected_team_id
        away_id = self.away_search.selected_team_id

        if home_id is None or away_id is None:
            QMessageBox.warning(
                self,
                "Selection error",
                "Please select both teams from the suggestions.",
            )
            return

        # UI feedback
        self.battle_btn.setEnabled(False)
        self.battle_btn.setText("Computing...")
        self._show_loading_dialog()

        # Create worker & thread
        thread = QThread(self)
        worker = PredictionWorker(home_id, away_id)
        worker.moveToThread(thread)

        # Connect signals
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_worker_finished)
        worker.error.connect(self._on_worker_error)

        # Thread cleanup
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self._clear_worker_refs)

        # Keep refs
        self._worker_thread = thread
        self._worker = worker

        # Start!
        thread.start()

    def _on_worker_finished(self, summary: Dict[str, Dict[str, float]]):
        self._hide_loading_dialog()
        self._update_stats_ui(summary)
        self.battle_btn.setEnabled(True)
        self.battle_btn.setText("Battle")

    def _on_worker_error(self, msg: str):
        self._hide_loading_dialog()
        QMessageBox.critical(
            self,
            "Prediction error",
            msg,
        )
        self.battle_btn.setEnabled(True)
        self.battle_btn.setText("Battle")

    # ---------- Entry point ----------


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
