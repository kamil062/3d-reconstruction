# -*- coding: utf-8 -*-

import cv2
import numpy as np
import sfm
import time
import vis
import wx
import wx.lib.scrolledpanel
from threading import Thread
from wx.lib.newevent import NewEvent

GaugeUpdate, EVT_GAUGE_UPDATE = NewEvent()
ListUpdate, EVT_LIST_UPDATE = NewEvent()


class Panel(wx.Panel):

    def __init__(self, p, color='#ededed', s=wx.NO_BORDER, sizer=wx.VERTICAL, padding=(1, 1)):
        super(Panel, self).__init__(parent=p, style=s)
        self.SetBackgroundColour(color)

        self.sizer = wx.BoxSizer(sizer)
        self.SetSizer(self.sizer)

        self.grid_sizer = wx.GridBagSizer(*padding)

    def Add(self, element, proportion, flags=0, border=0):
        self.sizer.Add(element, proportion, flags, border)
        self.SetSizer(self.sizer)

    def AddGrid(self, element, position, span=wx.DefaultSpan, flags=0, border=0):
        self.grid_sizer.Add(element, position, span, flags, border)
        self.SetSizerAndFit(self.grid_sizer)

    def AddGrowableCol(self, i):
        self.grid_sizer.AddGrowableCol(i)

    def AddGrowableRow(self, i):
        self.grid_sizer.AddGrowableRow(i)


class ScrolledPanel(wx.lib.scrolledpanel.ScrolledPanel):

    def __init__(self, p, color='#ededed', s=wx.NO_BORDER, sizer=wx.VERTICAL, padding=(1, 1)):
        super(ScrolledPanel, self).__init__(parent=p, style=s)

        self.SetBackgroundColour(color)

        self.sizer = wx.BoxSizer(sizer)
        self.SetSizer(self.sizer)

        self.grid_sizer = wx.GridBagSizer(*padding)

    def Add(self, element, proportion, flags=0, border=0):
        self.sizer.Add(element, proportion, flags, border)
        self.SetSizer(self.sizer)

    def AddGrid(self, element, position, span=wx.DefaultSpan, flags=0, border=0):
        self.grid_sizer.Add(element, position, span, flags, border)
        self.SetSizerAndFit(self.grid_sizer)

    def AddGrowableCol(self, i):
        self.grid_sizer.AddGrowableCol(i)

    def AddGrowableRow(self, i):
        self.grid_sizer.AddGrowableRow(i)


class GaugeThread(Thread):

    def __init__(self, panel, sfm):
        self.sfm = sfm
        self.panel = panel
        Thread.__init__(self)

    def run(self):

        while True:
            if self.sfm.progress < 1.0:
                time.sleep(0.1)
                wx.PostEvent(self.panel, GaugeUpdate(progress=int(self.sfm.progress * 100)))
            else:
                time.sleep(0.1)
                wx.PostEvent(self.panel, ListUpdate())
                break


class MainWindow(wx.Frame):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self._sfm = sfm.StructureFromMotion(np.arange(1, 10).reshape((3, 3)), np.zeros((1, 5)))

        self._init_window()
        self._make_menu()
        self._make_layout()

        self._in_progress = 0

    def _init_window(self):

        self.SetSize((1024, 800))
        self.SetTitle(u"Kamil Piec - Rekonstrukcja 3D na podstawie obrazów 2D")
        self.SetWindowStyle(wx.MINIMIZE_BOX | wx.SYSTEM_MENU | wx.CAPTION | wx.CLOSE_BOX)
        self.Centre()
        self.Show(True)

    def _make_menu(self):

        menubar = wx.MenuBar()
        file_menu = wx.Menu()

        self._end_item = wx.MenuItem(file_menu, wx.ID_EXIT, u'Zakończ', u'Zakończ aplikację')

        file_menu.AppendItem(self._end_item)

        menubar.Append(file_menu, '&Plik')

        self.CreateStatusBar()
        self.SetMenuBar(menubar)
        self.Bind(wx.EVT_MENU, self._on_quit, self._end_item)

    def _make_control_panel(self):

        self._add_image_button = wx.Button(self._control_panel, label=u"Dodaj obrazy")
        self._del_image_button = wx.Button(self._control_panel, label=u"Usuń obrazy")
        self._export_button = wx.Button(self._control_panel, label=u"Eksportuj obiekt")
        self._import2_button = wx.Button(self._control_panel, label=u"Importuj parametry kamery")
        self._export2_button = wx.Button(self._control_panel, label=u"Eksportuj parametry kamery")
        self._reconstruct_button = wx.Button(self._control_panel, label=u"Rekonstruuj")
        self._extract_button = wx.Button(self._control_panel, label=u"Wyodrębnij punkty kluczowe")
        self._visualize_button = wx.Button(self._control_panel, label=u"Pokaż wizualizację chmury punktów")

        self._add_image_button.Bind(wx.EVT_BUTTON, self._open_file)
        self._del_image_button.Bind(wx.EVT_BUTTON, self._remove_images)
        self._export_button.Bind(wx.EVT_BUTTON, self._export_object)
        self._import2_button.Bind(wx.EVT_BUTTON, self._import_parameters)
        self._export2_button.Bind(wx.EVT_BUTTON, self._export_parameters)
        self._reconstruct_button.Bind(wx.EVT_BUTTON, self._reconstruct)
        self._extract_button.Bind(wx.EVT_BUTTON, self._extract_keypoints)
        self._visualize_button.Bind(wx.EVT_BUTTON, self._visualize)

        self._control_panel.AddGrid(self._add_image_button, (0, 0), flags=wx.EXPAND | wx.ALL)
        self._control_panel.AddGrid(self._del_image_button, (0, 1), flags=wx.EXPAND | wx.ALL)
        self._control_panel.AddGrid(self._export_button, (1, 1), flags=wx.EXPAND | wx.ALL)
        self._control_panel.AddGrid(self._import2_button, (2, 0), flags=wx.EXPAND | wx.ALL)
        self._control_panel.AddGrid(self._export2_button, (2, 1), flags=wx.EXPAND | wx.ALL)
        self._control_panel.AddGrid(self._extract_button, (3, 0), span=(1, 2), flags=wx.EXPAND | wx.ALL, border=2)
        self._control_panel.AddGrid(self._reconstruct_button, (4, 0), span=(1, 2), flags=wx.EXPAND | wx.ALL, border=2)
        self._control_panel.AddGrid(self._visualize_button, (5, 0), span=(1, 2), flags=wx.EXPAND | wx.ALL, border=2)

        for i in range(2):
            self._control_panel.AddGrowableCol(i)

        for i in range(4):
            self._control_panel.AddGrowableRow(i)

    def _make_K_panel(self):

        self._K_label = wx.StaticText(self._K_panel,
                                      label=u"Macierz parametrów wewnętrznych kamery K:",
                                      style=wx.ALIGN_CENTER_HORIZONTAL | wx.ALIGN_CENTER_VERTICAL)
        self._K_panel.AddGrid(self._K_label, (0, 0), span=(1, 3), flags=wx.EXPAND | wx.ALL, border=1)

        self._K_val = []
        w, h = self._K_panel.GetSize()

        for _ in range(9):
            self._K_val.append(wx.SpinCtrlDouble(self._K_panel, value="1.0", min=0.0, max=10000.0, size=(w / 3, h / 4),
                                                 style=wx.TE_CENTER))

        for i in range(3):
            for j in range(3):
                self._K_panel.AddGrid(self._K_val[i * 3 + j], (i + 1, j), flags=wx.EXPAND | wx.ALL, border=1)

        for i in range(3):
            self._K_panel.AddGrowableCol(i)
        for i in range(4):
            self._K_panel.AddGrowableRow(i)

    def _make_d_panel(self):

        self._d_label = wx.StaticText(self._d_panel,
                                      label=u"Wektor współczynników dystorsji d:",
                                      style=wx.ALIGN_CENTER_HORIZONTAL | wx.ALIGN_CENTER_VERTICAL)
        self._d_panel.AddGrid(self._d_label, (0, 0), span=(1, 5), flags=wx.EXPAND | wx.ALL, border=1)

        self._d_val = []
        w, h = self._d_panel.GetSize()

        for i in range(5):
            self._d_val.append(wx.SpinCtrlDouble(self._d_panel, value="0.0", min=-100.0, max=100.0, size=(w / 5, 20),
                                                 style=wx.TE_CENTER))
            self._d_panel.AddGrid(self._d_val[i], (1, i), flags=wx.EXPAND | wx.ALL, border=1)

        for i in range(5):
            self._d_panel.AddGrowableCol(i)

    def _make_match_panel(self):

        self._feature_label = wx.StaticText(self._match_panel, label=u"Metoda dopasowywania cech")

        self._flann_radio = wx.RadioButton(self._match_panel, label='FLANN', style=wx.RB_GROUP)
        self._flann_label = wx.StaticText(self._match_panel, label=u"Ilość sprawdzeń:")
        self._flann_cb = wx.SpinCtrlDouble(self._match_panel, value="150", min=10, max=1000, inc=25)

        self._bf_radio = wx.RadioButton(self._match_panel, label='BFMatcher')
        self._bf_label = wx.StaticText(self._match_panel, label=u"Miara odległości:")
        self._bf_cb = wx.ComboBox(self._match_panel,
                                  choices=['Euklidesowa (NORM_L2)', 'Manhattan (NORM_L1)', 'Hamminga (NORM_HAMMING)'],
                                  style=wx.CB_READONLY)
        self._bf_cb.SetSelection(0)

        self._best_percent_label = wx.StaticText(self._match_panel,
                                                 label=u"Procent najlepszych punktów\nprzy dopasowywaniu cech:")
        self._best_percent = wx.SpinCtrl(self._match_panel, value="75", min=0, max=100)

        self._match_panel.AddGrid(self._feature_label, (0, 0), span=(1, 3), flags=wx.ALL | wx.ALIGN_CENTER_HORIZONTAL)
        self._match_panel.AddGrid(self._flann_radio, (1, 0), flags=wx.ALL | wx.ALIGN_CENTER_VERTICAL)
        self._match_panel.AddGrid(self._flann_label, (1, 1), flags=wx.ALL | wx.ALIGN_CENTER_VERTICAL, border=4)
        self._match_panel.AddGrid(self._flann_cb, (1, 2), flags=wx.EXPAND | wx.ALL)
        self._match_panel.AddGrid(self._bf_radio, (2, 0), flags=wx.ALL | wx.ALIGN_CENTER_VERTICAL)
        self._match_panel.AddGrid(self._bf_label, (2, 1), flags=wx.ALL | wx.ALIGN_CENTER_VERTICAL, border=4)
        self._match_panel.AddGrid(self._bf_cb, (2, 2), flags=wx.EXPAND | wx.ALL)
        self._match_panel.AddGrid(self._best_percent_label, (4, 0), span=(1, 2),
                                  flags=wx.ALL | wx.ALIGN_CENTER_VERTICAL)
        self._match_panel.AddGrid(self._best_percent, (4, 2), flags=wx.EXPAND | wx.ALL)

        for i in range(3):
            self._match_panel.AddGrowableCol(i)

    def _make_detect_panel(self):

        self._detect_label = wx.StaticText(self._detect_panel,
                                           label=u"Metoda wykrywania cech")

        self._sift_radio = wx.RadioButton(self._detect_panel, label='SIFT', style=wx.RB_GROUP)

        self._surf_radio = wx.RadioButton(self._detect_panel, label='SURF')
        self._surf_label = wx.StaticText(self._detect_panel,
                                         label=u"Próg hesjanu:")
        self._surf_cb = wx.SpinCtrlDouble(self._detect_panel, value="400", min=50, max=1000, inc=50)

        self._brief_radio = wx.RadioButton(self._detect_panel, label='BRIEF')

        self._orb_radio = wx.RadioButton(self._detect_panel, label='ORB')
        self._orb_label = wx.StaticText(self._detect_panel,
                                        label=u"Maksymalna ilość cech:")
        self._orb_cb = wx.SpinCtrlDouble(self._detect_panel, value="10000", min=0, max=100000, inc=500)

        self._fast_radio = wx.RadioButton(self._detect_panel, label='FAST')

        self._detect_panel.AddGrid(self._detect_label, (0, 0), span=(1, 3), flags=wx.ALL | wx.ALIGN_CENTER_HORIZONTAL)
        self._detect_panel.AddGrid(self._sift_radio, (1, 0), flags=wx.ALL | wx.ALIGN_CENTER_VERTICAL)
        self._detect_panel.AddGrid(self._surf_radio, (2, 0), flags=wx.ALL | wx.ALIGN_CENTER_VERTICAL)
        self._detect_panel.AddGrid(self._surf_label, (2, 1), flags=wx.ALL | wx.ALIGN_CENTER_VERTICAL, border=4)
        self._detect_panel.AddGrid(self._surf_cb, (2, 2), flags=wx.EXPAND | wx.ALL)
        self._detect_panel.AddGrid(self._brief_radio, (3, 0), flags=wx.ALL | wx.ALIGN_CENTER_VERTICAL)
        self._detect_panel.AddGrid(self._orb_radio, (4, 0), flags=wx.ALL | wx.ALIGN_CENTER_VERTICAL)
        self._detect_panel.AddGrid(self._orb_label, (4, 1), flags=wx.ALL | wx.ALIGN_CENTER_VERTICAL, border=4)
        self._detect_panel.AddGrid(self._orb_cb, (4, 2), flags=wx.EXPAND | wx.ALL)
        self._detect_panel.AddGrid(self._fast_radio, (5, 0), flags=wx.ALL | wx.ALIGN_CENTER_VERTICAL)

        for i in range(3):
            self._detect_panel.AddGrowableCol(i)

    def _make_others_panel(self):

        self._others_label = wx.StaticText(self._others_panel, label=u"Usuwanie punktów oddalonych")

        self._contamination_label = wx.StaticText(self._others_panel, label=u"Zanieczyszczenie:")
        self._contamination_cb = wx.SpinCtrlDouble(self._others_panel, value="0.1", min=0, max=0.5, inc=0.05)

        self._iqr_checkbox = wx.CheckBox(self._others_panel, label=u'Metoda międzykwartylowa (IQR)')
        self._iqr_min_label = wx.StaticText(self._others_panel, label=u"Q1:")
        self._iqr_min_cb = wx.SpinCtrlDouble(self._others_panel, value="0", min=0, max=100, inc=1)
        self._iqr_max_label = wx.StaticText(self._others_panel, label=u"Q3:")
        self._iqr_max_cb = wx.SpinCtrlDouble(self._others_panel, value="90", min=0, max=100, inc=1)

        self._lof_checkbox = wx.CheckBox(self._others_panel, label=u'Local Outlier Factor (LOF)')
        self._lof_label = wx.StaticText(self._others_panel, label=u"Ilość sąsiadów:")
        self._lof_cb = wx.SpinCtrlDouble(self._others_panel, value="20", min=1, max=1000, inc=10)

        self._forest_checkbox = wx.CheckBox(self._others_panel, label=u'Isolation forest')

        self._ee_checkbox = wx.CheckBox(self._others_panel, label=u'Elliptic Envelope')

        self._others_label2 = wx.StaticText(self._others_panel, label=u"Inne ustawienia")
        self._scale_label = wx.StaticText(self._others_panel, label=u"Skalowanie obrazów wejściowych:")
        self._scale_cb = wx.ComboBox(self._others_panel, choices=[str(i) for i in range(1, 9)], style=wx.CB_READONLY)
        self._scale_cb.SetSelection(0)

        self._others_panel.AddGrid(self._others_label, (0, 0), span=(1, 3), flags=wx.ALL | wx.ALIGN_CENTER_HORIZONTAL)

        self._others_panel.AddGrid(self._contamination_label, (1, 0), flags=wx.ALL | wx.ALIGN_CENTER_VERTICAL)
        self._others_panel.AddGrid(self._contamination_cb, (1, 1), span=(1, 2), flags=wx.EXPAND | wx.ALL)

        self._others_panel.AddGrid(self._iqr_checkbox, (2, 0), span=(2, 1), flags=wx.ALL | wx.ALIGN_CENTER_VERTICAL)
        self._others_panel.AddGrid(self._iqr_min_label, (2, 1), flags=wx.ALL | wx.ALIGN_CENTER_VERTICAL, border=4)
        self._others_panel.AddGrid(self._iqr_min_cb, (2, 2), flags=wx.EXPAND | wx.ALL)
        self._others_panel.AddGrid(self._iqr_max_label, (3, 1), flags=wx.ALL | wx.ALIGN_CENTER_VERTICAL, border=4)
        self._others_panel.AddGrid(self._iqr_max_cb, (3, 2), flags=wx.EXPAND | wx.ALL)

        self._others_panel.AddGrid(self._lof_checkbox, (4, 0), flags=wx.ALL | wx.ALIGN_CENTER_VERTICAL)
        self._others_panel.AddGrid(self._lof_label, (4, 1), flags=wx.ALL | wx.ALIGN_CENTER_VERTICAL, border=4)
        self._others_panel.AddGrid(self._lof_cb, (4, 2), flags=wx.EXPAND | wx.ALL)

        self._others_panel.AddGrid(self._forest_checkbox, (5, 0), flags=wx.ALL | wx.ALIGN_CENTER_VERTICAL)
        self._others_panel.AddGrid(self._ee_checkbox, (6, 0), flags=wx.ALL | wx.ALIGN_CENTER_VERTICAL)

        self._others_panel.AddGrid(self._others_label2, (7, 0), span=(1, 3), flags=wx.ALL | wx.ALIGN_CENTER_HORIZONTAL)
        self._others_panel.AddGrid(self._scale_label, (8, 0), flags=wx.ALL | wx.ALIGN_CENTER_VERTICAL)
        self._others_panel.AddGrid(self._scale_cb, (8, 1), span=(1, 2), flags=wx.EXPAND | wx.ALL)

        for i in range(2):
            self._others_panel.AddGrowableCol(i)

    def _make_layout(self):

        self._main_panel = Panel(self, sizer=wx.HORIZONTAL)

        self._left_panel = Panel(self._main_panel)
        self._list_panel = ScrolledPanel(self._left_panel, s=wx.BORDER_DOUBLE)
        self._list_panel.SetupScrolling(scroll_x=False)
        self._list_panel_inner = Panel(self._list_panel)
        self._image_panel = Panel(self._left_panel, s=wx.BORDER_DOUBLE)
        self._gauge = wx.Gauge(self._left_panel, range=100)

        self._right_panel = Panel(self._main_panel)
        self._match_panel = Panel(self._right_panel, s=wx.BORDER_DOUBLE)
        self._detect_panel = Panel(self._right_panel, s=wx.BORDER_DOUBLE)
        self._others_panel = Panel(self._right_panel, s=wx.BORDER_DOUBLE)
        self._K_panel = Panel(self._left_panel, s=wx.BORDER_DOUBLE)
        self._d_panel = Panel(self._left_panel, s=wx.BORDER_DOUBLE)
        self._control_panel = Panel(self._right_panel, s=wx.BORDER_DOUBLE)

        self._make_control_panel()
        self._make_K_panel()
        self._make_d_panel()
        self._make_match_panel()
        self._make_detect_panel()
        self._make_others_panel()

        self._list_panel.Add(self._list_panel_inner, 1, wx.EXPAND | wx.ALL, 2)

        self._left_panel.Add(self._list_panel, 7, wx.EXPAND | wx.ALL, 2)
        self._left_panel.Add(self._image_panel, 10, wx.EXPAND | wx.ALL, 2)
        self._left_panel.Add(self._K_panel, 4, wx.EXPAND | wx.ALL, 2)
        self._left_panel.Add(self._d_panel, 1, wx.EXPAND | wx.ALL, 2)
        self._left_panel.Add(self._gauge, 1, wx.EXPAND | wx.ALL, 2)

        self._right_panel.Add(self._control_panel, 2, wx.EXPAND | wx.ALL, 2)
        self._right_panel.Add(self._detect_panel, 3, wx.EXPAND | wx.ALL, 2)
        self._right_panel.Add(self._match_panel, 2, wx.EXPAND | wx.ALL, 2)
        self._right_panel.Add(self._others_panel, 4, wx.EXPAND | wx.ALL, 2)

        self._main_panel.Add(self._left_panel, 1, wx.EXPAND | wx.ALL, 2)
        self._main_panel.Add(self._right_panel, 1, wx.EXPAND | wx.ALL, 2)

        self.Bind(EVT_GAUGE_UPDATE, self._update_gauge)
        self.Bind(EVT_LIST_UPDATE, self._show_list)

    def _show_list(self, e):

        for child in self._list_panel_inner.GetChildren():
            child.Destroy()

        j = 1
        panel_w = self._list_panel.GetSize()[0] / 10

        for img in self._sfm.get_images():
            height, width = img.shape[:2]

            img = wx.ImageFromBuffer(width, height, img)
            bmp = wx.BitmapFromImage(img.Rescale(panel_w - 28, panel_w - 28))
            bmp2 = wx.EmptyBitmap(width=panel_w - 28, height=panel_w - 10)
            dc = wx.MemoryDC(bmp2)
            dc.SetBackground(wx.Brush('#ededed'))
            dc.Clear()
            dc.DrawBitmap(bmp, 0, 18)
            dc.SetTextForeground((0, 0, 0))
            dc.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
            tw, th = dc.GetTextExtent(str(j))
            dc.DrawText(str(j), (panel_w - 28) / 2 - tw / 2, 0)
            btn = wx.BitmapButton(self._list_panel_inner, bitmap=bmp2, size=wx.Size(panel_w - 3, panel_w))
            btn.img_id = j - 1

            self._list_panel_inner.AddGrid(btn, ((j - 1) / 10, (j - 1) % 10), flags=wx.EXPAND | wx.ALL)

            btn.Bind(wx.EVT_LEFT_DOWN, self._list_button_left)
            btn.Bind(wx.EVT_RIGHT_DOWN, self._list_button_right)

            j += 1

        self._list_panel_inner.Layout()
        self._list_panel_inner.Fit()
        self._list_panel.SetupScrolling(rate_y=1, scroll_x=False)
        self.Refresh()
        self._unlock_buttons()

    def _lock_buttons(self):

        self._add_image_button.Disable()
        self._del_image_button.Disable()
        self._export_button.Disable()
        self._import2_button.Disable()
        self._export2_button.Disable()
        self._reconstruct_button.Disable()
        self._extract_button.Disable()
        self._visualize_button.Disable()

    def _unlock_buttons(self):

        self._add_image_button.Enable()
        self._del_image_button.Enable()
        self._export_button.Enable()
        self._import2_button.Enable()
        self._export2_button.Enable()
        self._reconstruct_button.Enable()
        self._extract_button.Enable()
        self._visualize_button.Enable()

    def _run_threaded(self, fun, args):

        self._lock_buttons()

        thread2 = Thread(target=fun, args=args)
        thread2.daemon = True
        thread2.start()

        thread1 = GaugeThread(panel=self, sfm=self._sfm)
        thread1.daemon = True
        thread1.start()

    def _open_file(self, e):

        open_file_dialog = wx.FileDialog(self, u"Otwórz obraz", "", "",
                                         "Pliki graficzne (*.png;*.jpg)|*.png;*.jpg",
                                         wx.FD_OPEN | wx.FD_FILE_MUST_EXIST | wx.FD_MULTIPLE)

        if open_file_dialog.ShowModal() == wx.ID_CANCEL:
            return

        self._sfm.set_scale(int(self._scale_cb.GetStringSelection()))
        self._sfm.set_K(np.asarray([float(k.Value) for k in self._K_val]).reshape(3, 3))
        self._sfm.set_d(np.asarray([float(d.Value) for d in self._d_val]).reshape(1, 5))

        self._run_threaded(self._sfm.load_images, (open_file_dialog.GetPaths(),))

    def _list_button_left(self, e):

        img_id = e.GetEventObject().img_id

        img = self._sfm.get_images()[img_id]
        img = cv2.resize(img, (self._image_panel.GetSize()[0], self._image_panel.GetSize()[1]))
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        h, w = img.shape[:2]
        wxbmp = wx.BitmapFromBuffer(w, h, img)
        dc = wx.ClientDC(self._image_panel)
        dc.DrawBitmap(wxbmp, 0, 0)

    def _list_button_right(self, e):

        img_id = e.GetEventObject().img_id

        img = self._sfm.get_keypoint_images()[img_id]
        if img is not None:
            img = cv2.resize(img, (self._image_panel.GetSize()[0], self._image_panel.GetSize()[1]))
            if len(img.shape) > 2:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            h, w = img.shape[:2]
            wxbmp = wx.BitmapFromBuffer(w, h, img)
            dc = wx.ClientDC(self._image_panel)
            dc.DrawBitmap(wxbmp, 0, 0)

    def _remove_images(self, e):

        indices_dialog = wx.TextEntryDialog(self,
                                            u'Wpisz numery obrazów do usunięcia (rozdzielone przecinkami)',
                                            u'Usuwanie obrazów')

        if indices_dialog.ShowModal() == wx.ID_OK:
            indices_str = indices_dialog.GetValue().split(',')
            indices = []

            for i in indices_str:
                try:
                    indices.append(abs(int(float(i))))
                except ValueError:
                    pass

            self._run_threaded(self._sfm.remove_images, ([i - 1 for i in indices],))

        indices_dialog.Destroy()

        dial = wx.MessageDialog(self, u'Usunięto obrazy z listy', u'Informacja', wx.OK | wx.ICON_INFORMATION)
        dial.ShowModal()

    def _import_parameters(self, e):

        open_file_dialog = wx.FileDialog(self, u"Otwórz plik", "", "",
                                         "Pliki tekstowe (*.txt)|*.txt", wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)

        if open_file_dialog.ShowModal() == wx.ID_CANCEL:
            return

        with open(open_file_dialog.GetPath(), 'r') as f:
            params = np.loadtxt(f)

            for i in range(len(self._K_val)):
                self._K_val[i].SetValue(params[i])

            for i in range(len(self._d_val)):
                self._d_val[i].SetValue(params[9:][i])

            self._sfm.set_K(params[:9].reshape(3, 3))
            self._sfm.set_d(params[9:].reshape(1, 5))

        dial = wx.MessageDialog(self, u'Zaimportowano parametry', u'Informacja', wx.OK | wx.ICON_INFORMATION)
        dial.ShowModal()

    def _export_parameters(self, e):

        save_file_dialog = wx.FileDialog(self, u"Zapisz plik", "", "",
                                         "Pliki tekstowe (*.txt)|*.txt", wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)

        if save_file_dialog.ShowModal() == wx.ID_CANCEL:
            return

        with open(save_file_dialog.GetPath(), 'wb') as f:
            values = [float(k.Value) for k in self._K_val] + [float(d.Value) for d in self._d_val]
            np.savetxt(f, np.asarray(values))

        dial = wx.MessageDialog(self, u'Wyeksportowano parametry', u'Informacja', wx.OK | wx.ICON_INFORMATION)
        dial.ShowModal()

    def _extract_keypoints(self, e):

        global method
        surf_param = 400
        orb_param = 10000

        if self._sift_radio.GetValue():
            method = "sift"
        elif self._surf_radio.GetValue():
            method = "surf"
            surf_param = int(self._surf_cb.GetValue())
        elif self._brief_radio.GetValue():
            method = "brief"
        elif self._orb_radio.GetValue():
            method = "orb"
            orb_param = int(self._orb_cb.GetValue())
        elif self._fast_radio.GetValue():
            method = "fast"

        self._run_threaded(self._sfm.extract_keypoints, (method, surf_param, orb_param))

    def _reconstruct(self, e):

        global method
        flann_checks = 150
        bf_distance = cv2.NORM_L2

        if self._flann_radio.GetValue():
            method = "flann"
            flann_checks = int(self._flann_cb.GetValue())
        elif self._bf_radio.GetValue():
            method = "bf"
            if self._bf_cb.GetStringSelection() == "Euklidesowa (NORM_L2)":
                bf_distance = cv2.NORM_L2
            elif self._bf_cb.GetStringSelection() == "Manhattan (NORM_L1)":
                bf_distance = cv2.NORM_L1
            elif self._bf_cb.GetStringSelection() == "Hamminga (NORM_HAMMING)":
                bf_distance = cv2.NORM_HAMMING

        best_percent = int(self._best_percent.GetValue()) / 100.0

        self._run_threaded(self._sfm.reconstruct, (flann_checks, bf_distance, method, best_percent,))

    def _visualize(self, e):

        points = self._sfm.points_3d

        if self._iqr_checkbox.GetValue():
            Q1 = int(self._iqr_min_cb.GetValue())
            Q3 = int(self._iqr_max_cb.GetValue())

            points = vis.OutliersDetection.find_IQR(points, Q1, Q3)

        if self._lof_checkbox.GetValue():
            contamination = float(self._contamination_cb.GetValue())
            neighbors = int(self._lof_cb.GetValue())

            points = vis.OutliersDetection.find_LOF(points, contamination, neighbors)

        if self._forest_checkbox.GetValue():
            contamination = float(self._contamination_cb.GetValue())

            points = vis.OutliersDetection.find_isolation_forest(points, contamination)

        if self._ee_checkbox.GetValue():
            contamination = float(self._contamination_cb.GetValue())

            points = vis.OutliersDetection.find_elliptic_envelope(points, contamination)

        v = vis.Visualization(points)
        v.plot_point_cloud()

    def _export_object(self, e):

        save_file_dialog = wx.FileDialog(self, u"Zapisz plik", "", "",
                                         "Plik ply (*.ply)|*.ply", wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)

        if save_file_dialog.ShowModal() == wx.ID_CANCEL:
            return

        if self._sfm.points_3d is not None:
            points = self._sfm.points_3d
            points = np.concatenate((points[:, :3], points[:, 3:] * 255), axis=1)
            vis.Visualization.ply_save(save_file_dialog.GetPath(), points)

            dial = wx.MessageDialog(self, u'Wyeksportowano parametry', u'Informacja', wx.OK | wx.ICON_INFORMATION)
            dial.ShowModal()
        else:
            dial = wx.MessageDialog(self, u'Nie udało się wyeksportować parametrów', u'Informacja',
                                    wx.OK | wx.ICON_ERROR)
            dial.ShowModal()

    def _update_gauge(self, e):

        self._gauge.SetValue(int(self._sfm.progress * 100))

    def _on_quit(self, e):

        self.Close()


if __name__ == '__main__':
    app = wx.App()
    MainWindow(None)
    app.MainLoop()
