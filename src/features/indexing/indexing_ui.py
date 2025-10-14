import tkinter as tk
import threading
from tkinter import ttk, filedialog, messagebox, scrolledtext

from src.utils.logger import logger
from src.core.chunker import STRUCTURAL_PATTERNS_DEFAULT, SEMANTIC_PATTERNS_DEFAULT, SPECIAL_KEYWORDS_DEFAULT
from src.features.indexing.indexing_pipeline import run_indexing_pipeline
from src.utils.validators import validate_config

class IndexingTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent, padding="10")
        self.filepath = ""
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        self._create_file_management_widgets()
        self._create_chunking_settings_widgets()
        self._create_results_widgets()

    def _create_file_management_widgets(self):
        frame = ttk.LabelFrame(self, text="  📂 انتخاب و مدیریت فایل  ", padding="10")
        frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        frame.columnconfigure(1, weight=1)

        self.index_button = ttk.Button(frame, text="⚡️ ایندکس در Vector DB", command=self._start_indexing)
        self.index_button.grid(row=2, column=1, padx=5, pady=5, sticky='e')
        ttk.Button(frame, text="  انتخاب فایل Word  ", command=self._select_file).grid(row=0, column=0, padx=5, pady=5)
        self.file_label = ttk.Label(frame, text="📄 فایل: (هیچ فایلی انتخاب نشده)", anchor="w")
        self.file_label.grid(row=0, column=1, sticky="ew", padx=5)

    def _create_chunking_settings_widgets(self):
        frame = ttk.LabelFrame(self, text="  ⚙️ تنظیمات Chunking  ", padding="10")
        frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)

        params_frame = ttk.Frame(frame)
        params_frame.grid(row=0, column=0, sticky='ew', pady=5)
        # Simplified layout
        ttk.Label(params_frame, text="حداقل کلمات:").pack(side='left', padx=(5, 2))
        self.min_chunk_size = ttk.Spinbox(params_frame, from_=50, to=1000, increment=50, width=6); self.min_chunk_size.set("200"); self.min_chunk_size.pack(side='left', padx=2)
        ttk.Label(params_frame, text="حداکثر:").pack(side='left', padx=(10, 2))
        self.max_chunk_size = ttk.Spinbox(params_frame, from_=100, to=5000, increment=100, width=6); self.max_chunk_size.set("800"); self.max_chunk_size.pack(side='left', padx=2)
        ttk.Label(params_frame, text="همپوشانی:").pack(side='left', padx=(10, 2))
        self.overlap_size = ttk.Spinbox(params_frame, from_=0, to=500, increment=10, width=6); self.overlap_size.set("50"); self.overlap_size.pack(side='left', padx=2)

        notebook = ttk.Notebook(frame)
        notebook.grid(row=1, column=0, sticky='nsew', pady=10)
        self.struct_patterns_text = self._create_pattern_tab(notebook, "الگوهای ساختاری", STRUCTURAL_PATTERNS_DEFAULT)
        self.semantic_patterns_text = self._create_pattern_tab(notebook, "الگوهای معنایی", SEMANTIC_PATTERNS_DEFAULT)
        self.keywords_text = self._create_pattern_tab(notebook, "کلمات کلیدی خاص", SPECIAL_KEYWORDS_DEFAULT)

    def _create_pattern_tab(self, notebook, title, default_content):
        tab_frame = ttk.Frame(notebook, padding=5)
        notebook.add(tab_frame, text=title)
        tab_frame.rowconfigure(0, weight=1); tab_frame.columnconfigure(0, weight=1)
        text_area = scrolledtext.ScrolledText(tab_frame, wrap=tk.WORD, height=10, font=("Consolas", 9))
        text_area.insert(tk.END, default_content.strip())
        text_area.grid(row=0, column=0, sticky='nsew')
        ttk.Button(tab_frame, text="بازگشت به پیش‌فرض", command=lambda: text_area.delete(1.0, tk.END) or text_area.insert(tk.END, default_content.strip())).grid(row=1, column=0, sticky='e', pady=5)
        return text_area

    def _create_results_widgets(self):
        frame = ttk.LabelFrame(self, text="  📊 نتایج و گزارش  ", padding="10")
        frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        frame.columnconfigure(0, weight=1)
        self.progress_bar = ttk.Progressbar(frame, orient='horizontal', mode='determinate')
        self.progress_bar.grid(row=0, column=0, sticky='ew', pady=5)
        self.status_label = ttk.Label(frame, text="وضعیت: آماده برای شروع")
        self.status_label.grid(row=1, column=0, sticky='w', pady=2)
        self.log_text = scrolledtext.ScrolledText(frame, height=8, state='disabled', wrap=tk.WORD, font=("Segoe UI", 9))
        self.log_text.grid(row=2, column=0, sticky='ew', pady=5)

    def _select_file(self):
        fpath = filedialog.askopenfilename(title="انتخاب فایل Word", filetypes=(("Word Documents", "*.docx"), ("All files", "*.*")))
        if fpath:
            self.filepath = fpath
            self.file_label.config(text=f"📄 فایل: {self.filepath}")
            logger.info(f"File selected: {self.filepath}")

    def _get_current_config(self):
        return {
            "min_chunk_size": int(self.min_chunk_size.get()),
            "max_chunk_size": int(self.max_chunk_size.get()),
            "overlap_size": int(self.overlap_size.get()),
            "structural_patterns": self.struct_patterns_text.get(1.0, tk.END),
            "semantic_patterns": self.semantic_patterns_text.get(1.0, tk.END),
            "special_keywords": self.keywords_text.get(1.0, tk.END),
        }

    def _start_indexing(self):
        if not self.filepath:
            messagebox.showerror("خطا", "لطفاً ابتدا یک فایل Word را انتخاب کنید.")
            return

        config = self._get_current_config()
        is_valid, errors = validate_config(config)
        if not is_valid:
            messagebox.showerror("خطای تنظیمات", "\n".join(errors))
            return

        self.index_button.config(state="disabled")
        self._clear_results()

        callbacks = {
            'on_progress': lambda p, m: self.after(0, self._update_progress, p, m),
            'on_log': lambda m, l: self.after(0, self._add_log, m, l),
            'on_completion': lambda r: self.after(0, self._handle_completion, r),
        }

        thread = threading.Thread(target=run_indexing_pipeline, args=(self.filepath, config, callbacks))
        thread.daemon = True
        thread.start()

    def _clear_results(self):
        self.progress_bar['value'] = 0
        self.status_label.config(text="وضعیت: ...")
        self.log_text.config(state='normal')
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state='disabled')

    def _update_progress(self, percentage, message):
        self.progress_bar['value'] = percentage
        self.status_label.config(text=f"وضعیت: {message}")

    def _add_log(self, message, level):
        self.log_text.config(state='normal')
        # Simple coloring for different levels
        tag = f"log_{level}"
        self.log_text.tag_config(tag, foreground=self._get_color_for_level(level))
        self.log_text.insert(tk.END, f"{message}\n", tag)
        self.log_text.see(tk.END) # Scroll to the bottom
        self.log_text.config(state='disabled')

    def _get_color_for_level(self, level):
        if level == 'error': return 'red'
        if level == 'warning': return 'orange'
        return 'black'

    def _handle_completion(self, results):
        self.index_button.config(state="normal")
        final_message = f"✅ فرآیند با موفقیت در {results['total_time']} ثانیه کامل شد. کالکشن: {results['collection_name']}"
        self._add_log(final_message, 'info')
        messagebox.showinfo("تکمیل شد", final_message)

if __name__ == '__main__':
    # Helper for testing this tab directly
    root = tk.Tk()
    root.title("Test Indexing Tab")
    root.geometry("800x700")

    # Add project root to path for imports to work
    import sys, os
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, project_root)

    tab = IndexingTab(root)
    tab.pack(expand=True, fill='both')
    root.mainloop()