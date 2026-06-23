#!/usr/bin/env python3
"""
Calorie Tracker GUI — 6-Week Weight Loss Goal (900 kcal/day deficit)
Requires only the Python standard library (tkinter).
Data file: calorie_tracker_data.json  (back-compatible with all previous versions)
"""

import json
import os
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import date, datetime, timedelta

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calorie_tracker_data.json")

# ── Constants ─────────────────────────────────────────────────────────────────

KCAL_PER_LB          = 3500
GOAL_DEFICIT_PER_DAY = 500
GOAL_LBS             = 8.0    # target fat-loss in lbs
GOAL_WEEKS           = round(GOAL_LBS * KCAL_PER_LB / (GOAL_DEFICIT_PER_DAY * 7), 1)  # ~10

DAILY_FAT_ITEMS = [
    (0.014, "a sugar packet",          "sugar packets"),
    (0.057, "a AA battery",            "AA batteries"),
    (0.101, "a golf ball",             "golf balls"),
    (0.25,  "a stick of butter",       "sticks of butter"),
    (0.5,   "a baseball",              "baseballs"),
    (1.0,   "a can of soup",           "cans of soup"),
    (2.0,   "a large chicken breast",  "large chicken breasts"),
    (3.0,   "a hardcover book",        "hardcover books"),
    (5.0,   "a bag of flour",          "bags of flour"),
]

MILESTONES = {
    1:  ("1 lb of fat burned!",
         "That's like carrying one less can of soup around all day."),
    2:  ("2 lbs burned! Keep it up!",
         "Imagine two sticks of butter melted right off your body."),
    3:  ("3 lbs gone! Solid progress!",
         "You've shed the weight of a hardcover encyclopedia from pure effort."),
    4:  ("4 lbs down! Crushing it!",
         "That's the weight of a full laptop you've burned off. Incredible."),
    5:  ("5 lbs burned! Over halfway there!",
         "A full 5 lb bag of flour — gone. More than halfway to your goal!"),
    6:  ("6 lbs! Three quarters of the way!",
         "That's a full half-gallon of milk melted off your frame."),
    7:  ("7 lbs! One more to go!",
         "Imagine 7 sticks of butter — that's the fat you've torched."),
    8:  ("*** 8 LB GOAL CRUSHED! ***",
         "A full 8-lb bowling ball of fat, gone for good. You did it!"),
}

# ── Colour palette ────────────────────────────────────────────────────────────

BG       = "#1e1e2e"
SURFACE  = "#2a2a3e"
SURFACE2 = "#313244"
ACCENT   = "#cba6f7"
GREEN    = "#a6e3a1"
RED      = "#f38ba8"
YELLOW   = "#f9e2af"
TEXT     = "#cdd6f4"
SUBTEXT  = "#a6adc8"
BORDER   = "#45475a"

# ── Persistence ───────────────────────────────────────────────────────────────

def _migrate_profile(profile: dict) -> dict:
    """Inject fields added in later versions without overwriting existing data."""
    profile.setdefault("start_weight", profile.get("weight", 0))
    profile.setdefault("milestones_shown", [])
    if "goal_lbs" not in profile:
        # Old 5-lb CLI profiles had goal_weight = start_weight - 5.
        # Infer from stored values if present, otherwise default to current GOAL_LBS.
        stored_gw = profile.get("goal_weight")
        if stored_gw is not None:
            inferred = round(profile["start_weight"] - stored_gw, 1)
            # Only trust the inferred value if it looks like a real user-set goal (> 1 lb)
            profile["goal_lbs"] = inferred if inferred > 1 else GOAL_LBS
        else:
            profile["goal_lbs"] = GOAL_LBS
    if "deficit_target" not in profile:
        tdee   = profile.get("tdee", 0)
        budget = profile.get("daily_budget", 0)
        profile["deficit_target"] = (tdee - budget) if tdee and budget else 500
    if "budget_history" not in profile:
        # Restore original budget (TDEE - 500 old default) so all historical
        # days are measured against what was in effect when they were logged.
        tdee = profile.get("tdee", 0)
        original_budget = max(1200, round(tdee - 500)) if tdee > 0 else profile.get("daily_budget", 2998)
        profile["budget_history"] = [
            {"from": profile.get("created", "2000-01-01"), "budget": original_budget}
        ]
    return profile


def load_data() -> dict:
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            data = json.load(f)
        data.setdefault("log", {})
        data.setdefault("weight_log", {})   # kept in JSON for back-compat, not displayed
        if data.get("profile"):
            data["profile"] = _migrate_profile(data["profile"])
        return data
    return {"profile": None, "log": {}, "weight_log": {}}


def save_data(data: dict) -> None:
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)

# ── Calculations ──────────────────────────────────────────────────────────────

ACTIVITY_LEVELS = {
    "Sedentary":         ("Little or no exercise",              1.2),
    "Lightly active":    ("Light exercise 1-3 days/week",       1.375),
    "Moderately active": ("Moderate exercise 3-5 days/week",    1.55),
    "Very active":       ("Hard exercise 6-7 days/week",        1.725),
    "Extra active":      ("Very hard exercise or physical job", 1.9),
}


def calc_bmr(weight_lbs: float, height_in: float, age: int, sex: str) -> float:
    weight_kg = weight_lbs * 0.453592
    height_cm = height_in * 2.54
    if sex.lower() == "m":
        return 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    return 10 * weight_kg + 6.25 * height_cm - 5 * age - 161


def calc_tdee(bmr: float, act_mult: float) -> float:
    return bmr * act_mult


def calc_daily_budget(tdee: float, deficit: int = GOAL_DEFICIT_PER_DAY) -> int:
    return max(1200, round(tdee - deficit))


def calc_total_deficit_kcal(data: dict) -> float:
    """Sum deficits across all logged days using the budget in effect on each day."""
    profile  = data["profile"]
    history  = sorted(profile.get("budget_history", []), key=lambda x: x["from"])
    fallback = profile["daily_budget"]

    def budget_for(day_str: str) -> int:
        applicable = fallback
        for entry in history:
            if entry["from"] <= day_str:
                applicable = entry["budget"]
        return applicable

    total = 0.0
    for day_str, entries in data["log"].items():
        consumed = sum(e["calories"] for e in entries)
        total += budget_for(day_str) - consumed
    return total


def deficit_to_lbs(kcal: float) -> float:
    return kcal / KCAL_PER_LB


def fat_item_label(lbs: float) -> str:
    if lbs <= 0:
        return "nothing yet — stay in a deficit!"
    for threshold, singular, plural in DAILY_FAT_ITEMS:
        if lbs < threshold * 1.5:
            count = lbs / threshold
            return f"about {singular}" if count < 1.8 else f"about {count:.1f} {plural}"
    return f"about {lbs / 5.0:.1f} bags of flour"


def day_deficit_lbs(data: dict, day_key: str) -> float:
    budget   = data["profile"]["daily_budget"]
    entries  = data["log"].get(day_key, [])
    consumed = sum(e["calories"] for e in entries)
    return deficit_to_lbs(max(0, budget - consumed))


def today_key() -> str:
    return str(date.today())


def get_today_log(data: dict) -> list:
    return data["log"].setdefault(today_key(), [])

# ── Profile Dialog ────────────────────────────────────────────────────────────

class ProfileDialog(tk.Toplevel):
    def __init__(self, parent, data: dict, first_time: bool = False):
        super().__init__(parent)
        self.parent_app = parent
        self.data       = data
        self.result     = False
        self.first_time = first_time

        self.title("Profile Setup")
        self.configure(bg=BG)
        self.resizable(False, False)
        self.grab_set()

        self.geometry("460x590")
        self.update_idletasks()
        px = parent.winfo_x() + (parent.winfo_width()  - 460) // 2
        py = parent.winfo_y() + (parent.winfo_height() - 590) // 2
        self.geometry(f"460x590+{px}+{py}")

        if first_time:
            self.protocol("WM_DELETE_WINDOW", lambda: None)

        self._build(data.get("profile") or {})

    def _lbl(self, parent, text, row):
        tk.Label(parent, text=text, bg=BG, fg=SUBTEXT,
                 font=("Segoe UI", 10)).grid(row=row, column=0, sticky="w",
                                              padx=(0, 12), pady=5)

    def _entry_var(self, parent, row, default="") -> tk.StringVar:
        var = tk.StringVar(value=str(default))
        tk.Entry(parent, textvariable=var, bg=SURFACE2, fg=TEXT,
                 insertbackground=TEXT, relief="flat",
                 font=("Segoe UI", 10), highlightthickness=1,
                 highlightbackground=BORDER, highlightcolor=ACCENT
                 ).grid(row=row, column=1, sticky="ew", pady=5, ipady=3)
        return var

    def _build(self, prev: dict):
        tk.Label(self,
                 text="Welcome! Set up your profile" if self.first_time else "Edit Profile",
                 bg=BG, fg=ACCENT,
                 font=("Segoe UI", 14, "bold")).pack(pady=(20, 10))

        form = tk.Frame(self, bg=BG)
        form.pack(fill="both", expand=True, padx=24)
        form.columnconfigure(1, weight=1)

        self._lbl(form, "Name", 0)
        self.v_name = self._entry_var(form, 0, prev.get("name", ""))

        self._lbl(form, "Starting weight (lbs)", 1)
        self.v_weight = self._entry_var(
            form, 1, prev.get("start_weight") or prev.get("weight", ""))

        self._lbl(form, "Height", 2)
        hf = tk.Frame(form, bg=BG)
        hf.grid(row=2, column=1, sticky="ew", pady=5)
        prev_in = prev.get("height_in", 0)
        self.v_ft  = tk.StringVar(value=str(prev_in // 12) if prev_in else "")
        self.v_in2 = tk.StringVar(value=str(prev_in % 12)  if prev_in else "")
        for var, suffix, w in [(self.v_ft, " ft ", 4), (self.v_in2, " in", 4)]:
            tk.Entry(hf, textvariable=var, width=w, bg=SURFACE2, fg=TEXT,
                     insertbackground=TEXT, relief="flat", font=("Segoe UI", 10),
                     highlightthickness=1, highlightbackground=BORDER,
                     highlightcolor=ACCENT).pack(side="left")
            tk.Label(hf, text=suffix, bg=BG, fg=SUBTEXT,
                     font=("Segoe UI", 10)).pack(side="left")

        self._lbl(form, "Age", 3)
        self.v_age = self._entry_var(form, 3, prev.get("age", ""))

        self._lbl(form, "Sex", 4)
        self.v_sex = tk.StringVar(value=prev.get("sex", "m"))
        sf = tk.Frame(form, bg=BG)
        sf.grid(row=4, column=1, sticky="w", pady=5)
        for val, lbl in [("m", "Male"), ("f", "Female")]:
            tk.Radiobutton(sf, text=lbl, variable=self.v_sex, value=val,
                           bg=BG, fg=TEXT, selectcolor=SURFACE2,
                           activebackground=BG, activeforeground=ACCENT,
                           font=("Segoe UI", 10)).pack(side="left", padx=(0, 16))

        self._lbl(form, "Activity level", 5)
        acts = list(ACTIVITY_LEVELS.keys())
        self.v_act = tk.StringVar(value=prev.get("activity", acts[0]))
        ttk.Combobox(form, textvariable=self.v_act, values=acts,
                     state="readonly", font=("Segoe UI", 10)
                     ).grid(row=5, column=1, sticky="ew", pady=5)

        self.lbl_act_desc = tk.Label(form, text="", bg=BG, fg=SUBTEXT,
                                      font=("Segoe UI", 9, "italic"),
                                      wraplength=270, justify="left")
        self.lbl_act_desc.grid(row=6, column=0, columnspan=2, sticky="w", pady=(0, 8))

        def _upd(*_):
            desc, _ = ACTIVITY_LEVELS.get(self.v_act.get(), ("", 0))
            self.lbl_act_desc.config(text=desc)
        self.v_act.trace_add("write", _upd)
        _upd()

        bf = tk.Frame(self, bg=BG)
        bf.pack(pady=16)
        tk.Button(bf, text="Save Profile", command=self._save,
                  bg=ACCENT, fg=BG, font=("Segoe UI", 11, "bold"),
                  relief="flat", padx=20, pady=6, cursor="hand2").pack(side="left", padx=6)
        if not self.first_time:
            tk.Button(bf, text="Cancel", command=self.destroy,
                      bg=SURFACE2, fg=TEXT, font=("Segoe UI", 11),
                      relief="flat", padx=20, pady=6, cursor="hand2").pack(side="left", padx=6)

    def _save(self):
        try:
            name   = self.v_name.get().strip() or "User"
            weight = float(self.v_weight.get())
            ft     = int(self.v_ft.get())
            inch   = int(self.v_in2.get())
            age    = int(self.v_age.get())
        except ValueError:
            messagebox.showerror("Invalid input",
                                 "Please fill in all fields with valid numbers.", parent=self)
            return

        if weight <= 0 or ft < 1 or not (0 <= inch <= 11) or not (10 <= age <= 120):
            messagebox.showerror("Invalid input",
                                 "Check: weight > 0, valid height, age 10-120.", parent=self)
            return

        sex        = self.v_sex.get()
        act        = self.v_act.get()
        _, act_mult = ACTIVITY_LEVELS[act]
        total_in   = ft * 12 + inch
        bmr        = calc_bmr(weight, total_in, age, sex)
        tdee       = calc_tdee(bmr, act_mult)
        budget     = calc_daily_budget(tdee)

        prev        = self.data.get("profile") or {}
        prev_budget = prev.get("daily_budget", 0)

        # Carry forward budget_history; append new entry only if budget changed.
        hist = list(prev.get("budget_history") or [])
        if not hist:
            # First-time save: seed history from today
            hist = [{"from": str(date.today()), "budget": budget}]
        elif budget != prev_budget:
            hist.append({"from": str(date.today()), "budget": budget})

        self.data["profile"] = {
            "name":             name,
            "weight":           weight,
            "goal_weight":      round(weight - GOAL_LBS, 1),
            "goal_lbs":         GOAL_LBS,
            "deficit_target":   GOAL_DEFICIT_PER_DAY,
            "height_in":        total_in,
            "age":              age,
            "sex":              sex,
            "activity":         act,
            "act_mult":         act_mult,
            "bmr":              round(bmr),
            "tdee":             round(tdee),
            "daily_budget":     budget,
            "created":          prev.get("created", str(date.today())),
            "start_weight":     prev.get("start_weight", weight),
            "milestones_shown": prev.get("milestones_shown", []),
            "budget_history":   hist,
        }
        save_data(self.data)
        self.result = True
        self.destroy()

# ── Main Application ──────────────────────────────────────────────────────────

class CalorieTrackerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"Calorie Tracker — {GOAL_LBS} lb Goal")
        self.geometry("820x660")
        self.minsize(640, 520)
        self.configure(bg=BG)

        self.data     = load_data()
        self._pending: list[dict] = []   # items queued but not yet saved to log

        if self.data["profile"] is None:
            dlg = ProfileDialog(self, self.data, first_time=True)
            self.wait_window(dlg)

        self._configure_styles()
        self._build_ui()
        self._refresh_all()

    # ── ttk styles ────────────────────────────────────────────────────────────
    def _configure_styles(self):
        s = ttk.Style(self)
        s.theme_use("clam")

        s.configure(".",
                    background=BG, foreground=TEXT, font=("Segoe UI", 10))
        s.configure("TNotebook",     background=BG, borderwidth=0)
        s.configure("TNotebook.Tab", background=SURFACE, foreground=SUBTEXT,
                    padding=[16, 7], font=("Segoe UI", 10))
        s.map("TNotebook.Tab",
              background=[("selected", SURFACE2)],
              foreground=[("selected", ACCENT)])

        s.configure("TFrame",  background=BG)
        s.configure("TLabel",  background=BG, foreground=TEXT)

        s.configure("TProgressbar",
                    troughcolor=SURFACE2, background=ACCENT,
                    borderwidth=0, lightcolor=ACCENT, darkcolor=ACCENT)
        s.configure("Green.Horizontal.TProgressbar",
                    troughcolor=SURFACE2, background=GREEN,
                    borderwidth=0, lightcolor=GREEN, darkcolor=GREEN)
        s.configure("Red.Horizontal.TProgressbar",
                    troughcolor=SURFACE2, background=RED,
                    borderwidth=0, lightcolor=RED, darkcolor=RED)

        s.configure("Treeview",
                    background=SURFACE, foreground=TEXT,
                    fieldbackground=SURFACE, rowheight=30,
                    borderwidth=0, font=("Segoe UI", 10))
        s.configure("Treeview.Heading",
                    background=SURFACE2, foreground=ACCENT,
                    font=("Segoe UI", 10, "bold"), relief="flat")
        s.map("Treeview",
              background=[("selected", ACCENT)],
              foreground=[("selected", BG)])

        s.configure("TScrollbar",
                    background=SURFACE2, troughcolor=BG,
                    arrowcolor=SUBTEXT, borderwidth=0)
        s.configure("TCombobox",
                    fieldbackground=SURFACE2, background=SURFACE2,
                    foreground=TEXT, arrowcolor=ACCENT,
                    selectbackground=ACCENT, selectforeground=BG, borderwidth=0)
        s.map("TCombobox", fieldbackground=[("readonly", SURFACE2)])

    # ── UI skeleton ───────────────────────────────────────────────────────────
    def _build_ui(self):
        # Header bar
        hdr = tk.Frame(self, bg=SURFACE, pady=10)
        hdr.pack(fill="x")
        self.lbl_title = tk.Label(hdr, text="", bg=SURFACE, fg=ACCENT,
                                   font=("Segoe UI", 13, "bold"))
        self.lbl_title.pack(side="left", padx=18)
        self.lbl_hdr_stats = tk.Label(hdr, text="", bg=SURFACE, fg=TEXT,
                                       font=("Segoe UI", 10))
        self.lbl_hdr_stats.pack(side="right", padx=18)

        # Top calorie progress bar
        pf = tk.Frame(self, bg=BG)
        pf.pack(fill="x", padx=18, pady=(8, 0))
        tk.Label(pf, text="Today's calorie progress",
                 bg=BG, fg=SUBTEXT, font=("Segoe UI", 9)).pack(side="left")
        self.lbl_today_rem = tk.Label(pf, text="", bg=BG, fg=GREEN,
                                       font=("Segoe UI", 9, "bold"))
        self.lbl_today_rem.pack(side="right")
        self.today_pbar = ttk.Progressbar(self, length=100, mode="determinate",
                                           maximum=100, style="TProgressbar")
        self.today_pbar.pack(fill="x", padx=18, pady=(2, 8))

        # Notebook
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.tab_today    = ttk.Frame(self.nb)
        self.tab_progress = ttk.Frame(self.nb)
        self.tab_weekly   = ttk.Frame(self.nb)
        self.tab_profile  = ttk.Frame(self.nb)

        self.nb.add(self.tab_today,    text="  Today  ")
        self.nb.add(self.tab_progress, text="  Progress  ")
        self.nb.add(self.tab_weekly,   text="  Weekly  ")
        self.nb.add(self.tab_profile,  text="  Profile  ")

        self._build_today_tab()
        self._build_progress_tab()
        self._build_weekly_tab()
        self._build_profile_tab()

        self.nb.bind("<<NotebookTabChanged>>", lambda _: self._refresh_all())

    # ── Today tab ─────────────────────────────────────────────────────────────
    def _build_today_tab(self):
        t = self.tab_today
        t.columnconfigure(0, weight=1)
        t.rowconfigure(1, weight=1)

        # Input row
        inp = tk.Frame(t, bg=BG, pady=10)
        inp.grid(row=0, column=0, sticky="ew", padx=12)
        inp.columnconfigure(1, weight=1)

        tk.Label(inp, text="Food / Meal", bg=BG, fg=SUBTEXT,
                 font=("Segoe UI", 9)).grid(row=0, column=0, sticky="w", padx=(0, 6))
        self.entry_name = tk.Entry(inp, bg=SURFACE2, fg=TEXT, insertbackground=TEXT,
                                    relief="flat", font=("Segoe UI", 10),
                                    highlightthickness=1, highlightbackground=BORDER,
                                    highlightcolor=ACCENT)
        self.entry_name.grid(row=0, column=1, sticky="ew", padx=(0, 10), ipady=4)

        tk.Label(inp, text="kcal", bg=BG, fg=SUBTEXT,
                 font=("Segoe UI", 9)).grid(row=0, column=2, padx=(0, 4))
        self.entry_cal = tk.Entry(inp, bg=SURFACE2, fg=TEXT, insertbackground=TEXT,
                                   relief="flat", font=("Segoe UI", 10), width=7,
                                   highlightthickness=1, highlightbackground=BORDER,
                                   highlightcolor=ACCENT)
        self.entry_cal.grid(row=0, column=3, padx=(0, 10), ipady=4)

        tk.Button(inp, text="Add to list", command=self._queue_food,
                  bg=SURFACE2, fg=ACCENT, font=("Segoe UI", 10, "bold"),
                  relief="flat", padx=14, pady=3, cursor="hand2").grid(row=0, column=4)

        self.entry_name.bind("<Return>", lambda _: self.entry_cal.focus_set())
        self.entry_cal.bind("<Return>",  lambda _: self._queue_food())

        # Batch action row (Save / Discard)
        act = tk.Frame(t, bg=BG, pady=2)
        act.grid(row=0, column=0, sticky="e", padx=12, pady=(44, 0))

        self.btn_save_pending = tk.Button(
            act, text="Save all to log (0 items)", command=self._save_pending,
            bg=GREEN, fg=BG, font=("Segoe UI", 9, "bold"),
            relief="flat", padx=10, pady=2, cursor="hand2")
        self.btn_save_pending.grid(row=0, column=0, padx=(0, 6))

        self.btn_discard_pending = tk.Button(
            act, text="Discard pending", command=self._discard_pending,
            bg=SURFACE2, fg=YELLOW, font=("Segoe UI", 9),
            relief="flat", padx=10, pady=2, cursor="hand2")
        self.btn_discard_pending.grid(row=0, column=1)

        # Treeview
        tf = tk.Frame(t, bg=BG)
        tf.grid(row=1, column=0, sticky="nsew", padx=12)
        tf.columnconfigure(0, weight=1)
        tf.rowconfigure(0, weight=1)

        self.tree_today = ttk.Treeview(tf, columns=("time", "name", "calories"),
                                        show="headings", selectmode="browse")
        self.tree_today.heading("time",     text="Time")
        self.tree_today.heading("name",     text="Food / Meal")
        self.tree_today.heading("calories", text="kcal")
        self.tree_today.column("time",     width=70,  anchor="center", stretch=False)
        self.tree_today.column("name",     width=340, anchor="w")
        self.tree_today.column("calories", width=80,  anchor="e",     stretch=False)

        self.tree_today.tag_configure("pending",
                                       foreground=YELLOW,
                                       font=("Segoe UI", 10, "italic"))

        vsb = ttk.Scrollbar(tf, orient="vertical", command=self.tree_today.yview)
        self.tree_today.configure(yscrollcommand=vsb.set)
        self.tree_today.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")

        # Stats + delete button
        bot = tk.Frame(t, bg=BG, pady=6)
        bot.grid(row=2, column=0, sticky="ew", padx=12)
        bot.columnconfigure(0, weight=1)

        self.lbl_today_stats = tk.Label(bot, text="", bg=BG, fg=TEXT,
                                         font=("Segoe UI", 10))
        self.lbl_today_stats.grid(row=0, column=0, sticky="w")

        tk.Button(bot, text="Delete selected", command=self._delete_food,
                  bg=SURFACE2, fg=RED, font=("Segoe UI", 9),
                  relief="flat", padx=10, pady=3, cursor="hand2").grid(row=0, column=1)

        self.lbl_fat_today = tk.Label(t, text="", bg=BG, fg=YELLOW,
                                       font=("Segoe UI", 9, "italic"))
        self.lbl_fat_today.grid(row=3, column=0, sticky="w", padx=12, pady=(0, 8))

    # ── Progress tab ──────────────────────────────────────────────────────────
    def _build_progress_tab(self):
        t = self.tab_progress
        t.columnconfigure(0, weight=1)

        # Deficit card
        c1 = tk.Frame(t, bg=SURFACE, padx=18, pady=14)
        c1.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 6))
        c1.columnconfigure(0, weight=1)

        tk.Label(c1, text="Calorie Deficit — Fat Burned",
                 bg=SURFACE, fg=ACCENT,
                 font=("Segoe UI", 12, "bold")).grid(row=0, column=0, sticky="w")
        self.lbl_deficit_main = tk.Label(c1, text="", bg=SURFACE, fg=TEXT,
                                          font=("Segoe UI", 10))
        self.lbl_deficit_main.grid(row=1, column=0, sticky="w", pady=(4, 2))
        self.deficit_pbar = ttk.Progressbar(c1, length=100, mode="determinate",
                                             maximum=100,
                                             style="Green.Horizontal.TProgressbar")
        self.deficit_pbar.grid(row=2, column=0, sticky="ew", pady=(4, 4))
        self.lbl_equiv = tk.Label(c1, text="", bg=SURFACE, fg=YELLOW,
                                   font=("Segoe UI", 9, "italic"))
        self.lbl_equiv.grid(row=3, column=0, sticky="w")
        self.lbl_eta = tk.Label(c1, text="", bg=SURFACE, fg=SUBTEXT,
                                 font=("Segoe UI", 9))
        self.lbl_eta.grid(row=4, column=0, sticky="w", pady=(4, 0))

        # Milestones card
        c2 = tk.Frame(t, bg=SURFACE, padx=18, pady=14)
        c2.grid(row=1, column=0, sticky="ew", padx=12, pady=6)
        c2.columnconfigure(0, weight=1)

        tk.Label(c2, text="Milestones", bg=SURFACE, fg=ACCENT,
                 font=("Segoe UI", 12, "bold")).grid(row=0, column=0, sticky="w",
                                                     pady=(0, 8))
        self.milestone_grid = tk.Frame(c2, bg=SURFACE)
        self.milestone_grid.grid(row=1, column=0, sticky="ew")

    # ── Weekly tab ────────────────────────────────────────────────────────────
    def _build_weekly_tab(self):
        t = self.tab_weekly
        t.columnconfigure(0, weight=1)
        t.rowconfigure(0, weight=1)

        tf = tk.Frame(t, bg=BG)
        tf.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)
        tf.columnconfigure(0, weight=1)
        tf.rowconfigure(0, weight=1)

        cols = ("date", "consumed", "budget", "diff", "status")
        self.tree_weekly = ttk.Treeview(tf, columns=cols, show="headings",
                                         selectmode="none")
        for col, heading, w, anc in [
            ("date",     "Date",     130, "center"),
            ("consumed", "Consumed", 110, "e"),
            ("budget",   "Budget",   110, "e"),
            ("diff",     "Diff",      90, "e"),
            ("status",   "Status",   110, "center"),
        ]:
            self.tree_weekly.heading(col, text=heading)
            self.tree_weekly.column(col, width=w, anchor=anc,
                                    stretch=(col == "date"))

        self.tree_weekly.tag_configure("ok",   foreground=GREEN)
        self.tree_weekly.tag_configure("over", foreground=RED)
        self.tree_weekly.tag_configure("none", foreground=SUBTEXT)

        vsb = ttk.Scrollbar(tf, orient="vertical", command=self.tree_weekly.yview)
        self.tree_weekly.configure(yscrollcommand=vsb.set)
        self.tree_weekly.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")

        self.lbl_weekly_summary = tk.Label(t, text="", bg=BG, fg=SUBTEXT,
                                            font=("Segoe UI", 9))
        self.lbl_weekly_summary.grid(row=1, column=0, pady=(0, 8))

    # ── Profile tab ───────────────────────────────────────────────────────────
    def _build_profile_tab(self):
        t = self.tab_profile
        t.columnconfigure(0, weight=1)

        card = tk.Frame(t, bg=SURFACE, padx=24, pady=18)
        card.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 6))
        card.columnconfigure(1, weight=1)

        self._prof_vars = {}
        fields = [
            ("name",           "Name"),
            ("daily_budget",   "Daily Budget"),
            ("tdee",           "TDEE"),
            ("bmr",            "BMR"),
            ("activity",       "Activity Level"),
            ("goal_lbs",       "Fat-loss Goal"),
            ("deficit_target", "Daily Deficit Target"),
            ("created",        "Program Started"),
        ]
        for i, (key, lbl) in enumerate(fields):
            tk.Label(card, text=lbl, bg=SURFACE, fg=SUBTEXT,
                     font=("Segoe UI", 10)).grid(row=i, column=0, sticky="w", pady=4)
            v = tk.Label(card, text="", bg=SURFACE, fg=TEXT,
                         font=("Segoe UI", 10, "bold"))
            v.grid(row=i, column=1, sticky="w", padx=(24, 0), pady=4)
            self._prof_vars[key] = v

        tk.Button(t, text="Edit Profile", command=self._show_profile_dialog,
                  bg=ACCENT, fg=BG, font=("Segoe UI", 11, "bold"),
                  relief="flat", padx=20, pady=7, cursor="hand2").grid(
                  row=1, column=0, pady=14)

    # ── Refresh ───────────────────────────────────────────────────────────────
    def _refresh_all(self):
        if not self.data.get("profile"):
            return
        self._refresh_header()
        self._refresh_today()
        self._refresh_progress()
        self._refresh_weekly()
        self._refresh_profile_tab()

    def _refresh_header(self):
        p         = self.data["profile"]
        budget    = p["daily_budget"]
        consumed  = sum(e["calories"] for e in get_today_log(self.data))
        remaining = budget - consumed
        pct       = min(100, consumed / budget * 100) if budget else 0

        self.lbl_title.config(
            text=f"Calorie Tracker — {GOAL_LBS} lb Goal  |  {p['name']}")
        self.lbl_hdr_stats.config(text=f"{consumed} / {budget} kcal today")

        if remaining >= 0:
            self.lbl_today_rem.config(
                text=f"{remaining:+d} kcal remaining", fg=GREEN)
            self.today_pbar.configure(style="TProgressbar")
        else:
            self.lbl_today_rem.config(
                text=f"{-remaining} kcal OVER budget", fg=RED)
            self.today_pbar.configure(style="Red.Horizontal.TProgressbar")
        self.today_pbar["value"] = pct

    def _refresh_today(self):
        p         = self.data["profile"]
        entries   = get_today_log(self.data)
        budget    = p["daily_budget"]
        committed = sum(e["calories"] for e in entries)
        pending   = sum(e["calories"] for e in self._pending)
        total     = committed + pending
        remaining = budget - total

        for row in self.tree_today.get_children():
            self.tree_today.delete(row)

        for i, e in enumerate(entries):
            self.tree_today.insert("", "end", iid=f"c_{i}",
                values=(e["time"], e["name"], f"{e['calories']} kcal"))

        for i, e in enumerate(self._pending):
            self.tree_today.insert("", "end", iid=f"p_{i}", tags=("pending",),
                values=("--:--", f"{e['name']}  (pending)", f"{e['calories']} kcal"))

        # Update Save/Discard button labels
        n = len(self._pending)
        self.btn_save_pending.config(
            text=f"Save all to log ({n} item{'s' if n != 1 else ''})",
            bg=GREEN if n > 0 else SURFACE2,
            fg=BG if n > 0 else SUBTEXT)
        self.btn_discard_pending.config(
            text=f"Discard pending ({pending} kcal)" if n > 0 else "Discard pending")

        if n > 0:
            stats = (f"Logged: {committed} kcal   Pending: +{pending} kcal   "
                     f"Total if saved: {total} kcal   Remaining: {budget - total:+d} kcal")
            self.lbl_today_stats.config(text=stats,
                                        fg=RED if remaining < 0 else YELLOW)
        else:
            self.lbl_today_stats.config(
                text=f"Consumed: {committed} kcal   Budget: {budget} kcal   "
                     f"Remaining: {remaining:+d} kcal",
                fg=RED if remaining < 0 else TEXT)

        today_lbs = day_deficit_lbs(self.data, today_key())
        if today_lbs > 0:
            self.lbl_fat_today.config(
                text=f"Fat burned today (deficit): {today_lbs:.3f} lbs  --  "
                     f"{fat_item_label(today_lbs)}")
        else:
            self.lbl_fat_today.config(
                text="No fat deficit today (at or over budget).")

    def _refresh_progress(self):
        p           = self.data["profile"]
        goal_lbs    = p.get("goal_lbs", GOAL_LBS)
        deficit_tgt = p.get("deficit_target", GOAL_DEFICIT_PER_DAY)

        deficit_kcal = calc_total_deficit_kcal(self.data)
        deficit_lbs  = max(0.0, deficit_to_lbs(deficit_kcal))
        pct          = min(100.0, deficit_lbs / goal_lbs * 100) if goal_lbs else 0

        self.lbl_deficit_main.config(
            text=f"{deficit_lbs:.3f} / {goal_lbs} lbs burned   "
                 f"({pct:.1f}%)   Total deficit: {deficit_kcal:,.0f} kcal")
        self.deficit_pbar["value"] = pct
        self.lbl_equiv.config(text=f"That's like: {fat_item_label(deficit_lbs)}")

        if deficit_lbs >= goal_lbs:
            self.lbl_eta.config(text=">> GOAL REACHED! <<", fg=GREEN)
        else:
            lbs_left   = goal_lbs - deficit_lbs
            weeks_left = lbs_left * KCAL_PER_LB / (deficit_tgt * 7)
            self.lbl_eta.config(
                text=f"{lbs_left:.2f} lbs remaining  "
                     f"(~{weeks_left:.1f} weeks at {deficit_tgt} kcal/day)",
                fg=SUBTEXT)

        # Rebuild milestone grid
        shown    = p.get("milestones_shown", [])
        relevant = {k: v for k, v in MILESTONES.items() if k <= goal_lbs + 1}
        for w in self.milestone_grid.winfo_children():
            w.destroy()

        COLS = 3
        for idx, (lb_mark, (headline, comparison)) in enumerate(relevant.items()):
            reached = deficit_lbs >= lb_mark
            cell_bg = SURFACE2 if reached else SURFACE
            icon_fg = GREEN    if reached else SUBTEXT
            icon    = "[+]"    if reached else "[ ]"

            cell = tk.Frame(self.milestone_grid, bg=cell_bg, padx=8, pady=6,
                            highlightthickness=1, highlightbackground=BORDER)
            cell.grid(row=idx // COLS, column=idx % COLS,
                      padx=4, pady=4, sticky="ew")
            self.milestone_grid.columnconfigure(idx % COLS, weight=1)

            tk.Label(cell, text=f"{icon}  {lb_mark} lb",
                     bg=cell_bg, fg=icon_fg,
                     font=("Segoe UI", 10, "bold")).pack(anchor="w")
            tk.Label(cell, text=comparison, bg=cell_bg,
                     fg=TEXT if reached else SUBTEXT,
                     font=("Segoe UI", 8), wraplength=175, justify="left").pack(anchor="w")

    def _refresh_weekly(self):
        p      = self.data["profile"]
        log    = self.data["log"]
        budget = p["daily_budget"]

        for row in self.tree_weekly.get_children():
            self.tree_weekly.delete(row)

        week_deficit = 0
        logged_days  = 0
        today        = date.today()
        for i in range(6, -1, -1):
            d        = str(today - timedelta(days=i))
            entries  = log.get(d, [])
            consumed = sum(e["calories"] for e in entries)

            if consumed == 0 and d not in log:
                self.tree_weekly.insert("", "end",
                    values=(d, "—", f"{budget} kcal", "—", "—"),
                    tags=("none",))
            else:
                diff = consumed - budget
                week_deficit += (budget - consumed)
                logged_days  += 1
                status = "On track" if diff <= 0 else "Over budget"
                tag    = "ok"       if diff <= 0 else "over"
                self.tree_weekly.insert("", "end",
                    values=(d, f"{consumed} kcal", f"{budget} kcal",
                            f"{diff:+d} kcal", status), tags=(tag,))

        if logged_days:
            fat_w = deficit_to_lbs(max(0, week_deficit))
            self.lbl_weekly_summary.config(
                text=f"7-day deficit: {week_deficit:+,.0f} kcal  "
                     f"≈ {fat_w:.3f} lbs fat burned this week")
        else:
            self.lbl_weekly_summary.config(text="No entries logged this week yet.")

    def _refresh_profile_tab(self):
        p = self.data.get("profile")
        if not p:
            return
        self._prof_vars["name"].config(text=p.get("name", ""))
        self._prof_vars["daily_budget"].config(
            text=f"{p.get('daily_budget', '')} kcal/day")
        self._prof_vars["tdee"].config(text=f"{p.get('tdee', '')} kcal")
        self._prof_vars["bmr"].config(text=f"{p.get('bmr', '')} kcal")
        self._prof_vars["activity"].config(text=p.get("activity", ""))
        self._prof_vars["goal_lbs"].config(
            text=f"{p.get('goal_lbs', GOAL_LBS)} lbs")
        self._prof_vars["deficit_target"].config(
            text=f"{p.get('deficit_target', GOAL_DEFICIT_PER_DAY)} kcal/day")
        self._prof_vars["created"].config(text=p.get("created", ""))

    # ── Actions ───────────────────────────────────────────────────────────────
    def _queue_food(self):
        """Add item to the pending staging list (does NOT write to disk)."""
        name    = self.entry_name.get().strip()
        cal_str = self.entry_cal.get().strip()

        if not name:
            messagebox.showwarning("Missing field",
                                   "Please enter a food name.", parent=self)
            return
        if not cal_str.isdigit() or int(cal_str) <= 0:
            messagebox.showwarning("Invalid calories",
                                   "Calories must be a positive whole number.", parent=self)
            return

        self._pending.append({"name": name, "calories": int(cal_str)})
        self.entry_name.delete(0, "end")
        self.entry_cal.delete(0, "end")
        self.entry_name.focus_set()
        self._refresh_today()

    def _save_pending(self):
        """Commit all pending items to today's log and save to disk."""
        if not self._pending:
            messagebox.showinfo("Nothing to save",
                                "Add some items to the list first.", parent=self)
            return

        log = get_today_log(self.data)
        now_time = datetime.now().strftime("%H:%M")
        for item in self._pending:
            log.append({
                "time":     now_time,
                "name":     item["name"],
                "calories": item["calories"],
            })
        self._pending.clear()
        save_data(self.data)
        self._check_milestones()
        self._refresh_all()

    def _discard_pending(self):
        """Discard the pending staging list without saving."""
        if not self._pending:
            return
        self._pending.clear()
        self._refresh_today()

    def _delete_food(self):
        sel = self.tree_today.selection()
        if not sel:
            messagebox.showinfo("Nothing selected",
                                "Select a food entry first.", parent=self)
            return
        iid = sel[0]
        if iid.startswith("p_"):
            # Remove from pending list
            idx = int(iid[2:])
            if 0 <= idx < len(self._pending):
                self._pending.pop(idx)
            self._refresh_today()
        elif iid.startswith("c_"):
            # Remove from committed log
            idx     = int(iid[2:])
            entries = get_today_log(self.data)
            if 0 <= idx < len(entries):
                entries.pop(idx)
                save_data(self.data)
                self._refresh_all()
        else:
            # Fallback for tree rows without prefixed iid (should not happen)
            idx     = self.tree_today.index(iid)
            entries = get_today_log(self.data)
            committed_count = len(entries)
            if idx < committed_count:
                entries.pop(idx)
                save_data(self.data)
                self._refresh_all()
            else:
                pending_idx = idx - committed_count
                if 0 <= pending_idx < len(self._pending):
                    self._pending.pop(pending_idx)
                self._refresh_today()

    def _check_milestones(self):
        p     = self.data["profile"]
        shown = p.setdefault("milestones_shown", [])
        lbs   = deficit_to_lbs(calc_total_deficit_kcal(self.data))

        new_hits = []
        for lb_mark, (headline, comparison) in MILESTONES.items():
            if lbs >= lb_mark and lb_mark not in shown:
                new_hits.append((headline, comparison))
                shown.append(lb_mark)

        if new_hits:
            save_data(self.data)
            for headline, comparison in new_hits:
                messagebox.showinfo(f"  {headline}", comparison, parent=self)

    def _show_profile_dialog(self):
        dlg = ProfileDialog(self, self.data, first_time=False)
        self.wait_window(dlg)
        if dlg.result:
            self._refresh_all()

# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    try:
        app = CalorieTrackerApp()
        app.mainloop()
    except tk.TclError as exc:
        print(f"Could not start GUI: {exc}")
        print("Ensure python3-tk is installed:  sudo apt install python3-tk")


if __name__ == "__main__":
    main()
