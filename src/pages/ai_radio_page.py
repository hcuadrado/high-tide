# ai_radio_page.py
#
# SPDX-License-Identifier: GPL-3.0-or-later

import threading
import logging
from gettext import gettext as _

from gi.repository import Adw, Gdk, GLib, GObject, Gtk

from .page import Page
from ..lib import utils
from ..widgets import HTAutoLoadWidget

logger = logging.getLogger(__name__)


class HTAIRadioPage(Page):
    """Single persistent AI Radio page — setup, prompt input, loading, and results."""

    __gtype_name__ = "HTAIRadioPage"

    __gsignals__ = {
        "generate": (GObject.SignalFlags.RUN_FIRST, None, (str,)),
        "refine": (GObject.SignalFlags.RUN_FIRST, None, (str, GObject.TYPE_PYOBJECT)),
        "cancel-generate": (GObject.SignalFlags.RUN_FIRST, None, ()),
    }

    def __init__(self) -> None:
        super().__init__()
        self.set_title(_("AI Radio"))
        self.set_tag("ai_radio")

        self._history: list = []
        self._tracks: list = []
        self._ai_title: str = ""
        self._initial_state: str = "prompt"
        self._pre_loading_state: str = "prompt"

        self._state_stack: Gtk.Stack | None = None
        self._prompt_entry: Adw.EntryRow | None = None
        self._chips_box: Gtk.Box | None = None
        self._chip_signals: list = []
        self._auto_load: HTAutoLoadWidget | None = None
        self._refine_entry: Adw.EntryRow | None = None
        self._save_button: Gtk.Button | None = None
        self._bottom_bar: Gtk.ActionBar | None = None

        # Must be added before the page is rendered to avoid layout loops.
        self._bottom_bar = self._build_bottom_bar()
        self.object.add_bottom_bar(self._bottom_bar)

    def disconnect_all(self) -> None:
        for chip, handler_id in self._chip_signals:
            chip.disconnect(handler_id)
        self._chip_signals.clear()
        super().disconnect_all()

    def _load_async(self) -> None:
        pass

    def _load_finish(self) -> None:
        self._state_stack = Gtk.Stack(
            transition_type=Gtk.StackTransitionType.CROSSFADE,
            vexpand=True,
            vhomogeneous=False,
        )
        self._state_stack.add_named(self._build_setup_view(), "setup")
        self._state_stack.add_named(self._build_prompt_view(), "prompt")
        self._state_stack.add_named(self._build_loading_view(), "loading")
        self._state_stack.add_named(self._build_results_view(), "results")
        self._state_stack.set_visible_child_name(self._initial_state)
        self.append(self._state_stack)

    #
    #   VIEW BUILDERS
    #

    def _build_setup_view(self) -> Gtk.Widget:
        prefs_btn = Gtk.Button(
            label=_("Open Preferences"),
            halign=Gtk.Align.CENTER,
            css_classes=["pill", "suggested-action"],
        )
        self.signals.append((prefs_btn, prefs_btn.connect(
            "clicked", lambda *_: self.activate_action("app.preferences", None)
        )))
        return Adw.StatusPage(
            title=_("AI Radio"),
            description=_(
                "Configure an AI provider in Preferences to generate "
                "personalized radio stations."
            ),
            icon_name="starred-symbolic",
            child=prefs_btn,
            vexpand=True,
            valign=Gtk.Align.CENTER,
        )

    def _build_prompt_view(self) -> Gtk.Widget:
        box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=12,
            margin_start=16,
            margin_end=16,
            margin_top=32,
            margin_bottom=24,
            halign=Gtk.Align.CENTER,
            hexpand=True,
            valign=Gtk.Align.CENTER,
            vexpand=True,
        )
        box.set_size_request(520, -1)

        self._prompt_entry = Adw.EntryRow(
            title=_("Describe the music you want…"),
        )
        prompt_list = Gtk.ListBox(
            selection_mode=Gtk.SelectionMode.NONE,
            css_classes=["boxed-list"],
        )
        prompt_list.append(self._prompt_entry)

        generate_btn = Gtk.Button(
            label=_("Generate"),
            halign=Gtk.Align.CENTER,
            css_classes=["pill", "suggested-action"],
            sensitive=False,
        )
        self.signals.append((generate_btn, generate_btn.connect(
            "clicked", self._on_generate_clicked
        )))
        self.signals.append((self._prompt_entry, self._prompt_entry.connect(
            "notify::text",
            lambda *_: generate_btn.set_sensitive(
                len(self._prompt_entry.get_text().strip()) > 0
            ),
        )))
        key_ctrl = Gtk.EventControllerKey()
        key_ctrl.set_propagation_phase(Gtk.PropagationPhase.CAPTURE)
        self.signals.append((key_ctrl, key_ctrl.connect(
            "key-pressed",
            lambda ctrl, keyval, _kc, _st: (
                self._on_generate_clicked() or True
            ) if keyval in (Gdk.KEY_Return, Gdk.KEY_KP_Enter) else False,
        )))
        self._prompt_entry.add_controller(key_ctrl)

        box.append(prompt_list)
        box.append(generate_btn)
        return box

    def _build_loading_view(self) -> Gtk.Widget:
        box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=18,
            halign=Gtk.Align.CENTER,
            valign=Gtk.Align.CENTER,
            vexpand=True,
        )
        box.append(Adw.Spinner())
        box.append(Gtk.Label(
            label=_("Generating your radio station…"),
            css_classes=["dim-label"],
        ))
        cancel_btn = Gtk.Button(
            label=_("Cancel"),
            halign=Gtk.Align.CENTER,
            css_classes=["pill"],
        )
        self.signals.append((cancel_btn, cancel_btn.connect(
            "clicked", self._on_cancel_clicked
        )))
        box.append(cancel_btn)
        return box

    def _build_results_view(self) -> Gtk.Widget:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)

        self._chips_box = Gtk.FlowBox(
            selection_mode=Gtk.SelectionMode.NONE,
            column_spacing=6,
            row_spacing=6,
            max_children_per_line=100,
            margin_start=12,
            margin_end=12,
            margin_top=8,
            margin_bottom=4,
        )
        box.append(self._chips_box)

        self._auto_load = HTAutoLoadWidget()
        self._auto_load.set_scrolled_window(self.scrolled_window)
        self.disconnectables.append(self._auto_load)
        box.append(self._auto_load)

        return box

    def _build_bottom_bar(self) -> Gtk.ActionBar:
        bar = Gtk.ActionBar(visible=False)

        self._refine_entry = Adw.EntryRow(
            title=_("Refine: make it more upbeat…"),
            hexpand=True,
        )
        refine_list = Gtk.ListBox(
            selection_mode=Gtk.SelectionMode.NONE,
            css_classes=["boxed-list"],
            hexpand=True,
            margin_start=6,
            margin_end=6,
        )
        refine_list.append(self._refine_entry)

        refine_btn = Gtk.Button(
            icon_name="go-next-symbolic",
            sensitive=False,
            valign=Gtk.Align.CENTER,
            css_classes=["flat", "circular"],
        )
        self.signals.append((refine_btn, refine_btn.connect(
            "clicked", lambda *_: self._on_refine_submit()
        )))
        self.signals.append((self._refine_entry, self._refine_entry.connect(
            "notify::text",
            lambda *_: refine_btn.set_sensitive(
                len(self._refine_entry.get_text().strip()) > 0
            ),
        )))
        refine_key_ctrl = Gtk.EventControllerKey()
        refine_key_ctrl.set_propagation_phase(Gtk.PropagationPhase.CAPTURE)
        self.signals.append((refine_key_ctrl, refine_key_ctrl.connect(
            "key-pressed",
            lambda ctrl, keyval, _kc, _st: (
                self._on_refine_submit() or True
            ) if keyval in (Gdk.KEY_Return, Gdk.KEY_KP_Enter) else False,
        )))
        self._refine_entry.add_controller(refine_key_ctrl)

        center_box = Gtk.Box(spacing=6, hexpand=True)
        center_box.append(refine_list)
        center_box.append(refine_btn)
        bar.set_center_widget(center_box)

        self._save_button = Gtk.Button(
            label=_("Save"),
            valign=Gtk.Align.CENTER,
            css_classes=["suggested-action", "pill"],
            margin_start=6,
        )
        self.signals.append((self._save_button, self._save_button.connect(
            "clicked", self._on_save_clicked
        )))
        bar.pack_end(self._save_button)

        return bar

    #
    #   EVENT HANDLERS
    #

    def _on_generate_clicked(self, *args) -> None:
        prompt = self._prompt_entry.get_text().strip()
        if not prompt:
            return
        self.set_loading(True)
        self.emit("generate", prompt)

    def _on_cancel_clicked(self, *args) -> None:
        self.emit("cancel-generate")
        self.set_loading(False)

    def _on_refine_submit(self) -> None:
        text = self._refine_entry.get_text().strip()
        if not text:
            return
        self.set_loading(True)
        self.emit("refine", text, self._history)

    def _populate_chips(self, suggestions: list) -> None:
        for chip, handler_id in self._chip_signals:
            chip.disconnect(handler_id)
        self._chip_signals.clear()
        while self._chips_box.get_first_child():
            self._chips_box.remove(self._chips_box.get_first_child())
        for suggestion in suggestions:
            chip = Gtk.Button(label=suggestion, css_classes=["pill"])
            handler_id = chip.connect("clicked", self._on_chip_clicked, suggestion)
            self._chip_signals.append((chip, handler_id))
            self._chips_box.append(chip)

    def _on_chip_clicked(self, btn, suggestion: str) -> None:
        self._refine_entry.set_text(suggestion)
        self._on_refine_submit()

    #
    #   SAVE AS PLAYLIST
    #

    def _on_save_clicked(self, *args) -> None:
        self._save_button.set_sensitive(False)
        self._save_button.set_child(Gtk.Spinner(spinning=True))
        threading.Thread(target=self._th_save_as_playlist).start()

    def _th_save_as_playlist(self) -> None:
        total = len(self._tracks)
        success_count = 0
        try:
            playlist = utils.session.user.create_playlist(
                self._ai_title, _("Generated by AI Radio")
            )
            for track in self._tracks:
                try:
                    playlist.add([track.id])
                    success_count += 1
                except Exception:
                    logger.exception("Failed to add track %s to playlist", track.id)
        except Exception:
            logger.exception("Failed to create AI Radio playlist")
            GLib.idle_add(self._on_save_complete, 0, total)
            return
        GLib.idle_add(self._on_save_complete, success_count, total)

    def _on_save_complete(self, success_count: int, total: int) -> None:
        self._save_button.set_sensitive(True)
        self._save_button.set_child(None)
        self._save_button.set_label(_("Save"))
        if success_count == total and total > 0:
            utils.send_toast(_("Saved as playlist"), 3)
        elif success_count > 0:
            utils.send_toast(
                _("Saved playlist with {} of {} tracks").format(success_count, total), 4
            )
        else:
            utils.send_toast(_("Could not save playlist — try again"), 3)

    #
    #   PUBLIC API
    #

    def show_setup_state(self) -> None:
        # Don't wipe results if the user already generated a radio.
        current = (
            self._state_stack.get_visible_child_name()
            if self._state_stack
            else self._initial_state
        )
        if current == "results":
            return
        self._initial_state = "setup"
        if self._state_stack:
            self._state_stack.set_visible_child_name("setup")
        if self._bottom_bar:
            self._bottom_bar.set_visible(False)

    def show_prompt_state(self) -> None:
        # Don't reset to prompt when results are already showing.
        current = (
            self._state_stack.get_visible_child_name()
            if self._state_stack
            else self._initial_state
        )
        if current == "results":
            return
        self._initial_state = "prompt"
        if self._state_stack:
            self._state_stack.set_visible_child_name("prompt")
        if self._bottom_bar:
            self._bottom_bar.set_visible(False)

    def set_loading(self, loading: bool) -> None:
        if not self._state_stack:
            return
        if loading:
            self._pre_loading_state = self._state_stack.get_visible_child_name()
            self._state_stack.set_visible_child_name("loading")
            if self._bottom_bar:
                self._bottom_bar.set_visible(False)
            if self.scrolled_window:
                self.scrolled_window.get_vadjustment().set_value(0)
        else:
            target = self._pre_loading_state if self._tracks else "prompt"
            self._state_stack.set_visible_child_name(target)
            if self._bottom_bar:
                self._bottom_bar.set_visible(target == "results")

    def update_tracks(
        self,
        title: str,
        tracks: list,
        suggestions: list,
        history: list,
    ) -> None:
        self._ai_title = title
        self._tracks = tracks
        self._history = history
        self.set_title(title)

        if self._auto_load:
            self._auto_load.set_items(tracks)
            self._auto_load.set_function(None)
        if self._chips_box:
            self._populate_chips(suggestions)
        if self._refine_entry:
            self._refine_entry.set_text("")

        # Delay showing results until HTAutoLoadWidget._add() idle callback fires,
        # so track widgets are populated before the view becomes visible.
        def _show_results():
            if self._state_stack:
                self._state_stack.set_visible_child_name("results")
            if self._bottom_bar:
                self._bottom_bar.set_visible(True)

        GLib.idle_add(_show_results)
