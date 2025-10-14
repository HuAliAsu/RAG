import tkinter as tk
import threading
from tkinter import ttk, messagebox

from src.utils.logger import logger
from src.features.searching.searching_logic import search_logic

class SearchingTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent, padding="10")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        self._create_search_input_widgets()
        self._create_search_results_widgets()

        # Add a refresh button or hook to notebook tab change
        self.refresh_button = ttk.Button(self, text="🔄 تازه‌سازی لیست پایگاه داده", command=self.populate_collections_dropdown)
        self.refresh_button.grid(row=2, column=0, sticky='w', padx=5, pady=10)

        self.populate_collections_dropdown()

    def _create_search_input_widgets(self):
        frame = ttk.LabelFrame(self, text="  🔎 جستجوی معنایی  ", padding="10")
        frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="پایگاه داده:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.collection_var = tk.StringVar()
        self.collection_dropdown = ttk.Combobox(frame, textvariable=self.collection_var, state='readonly')
        self.collection_dropdown.grid(row=0, column=1, sticky='ew', padx=5, pady=5)

        ttk.Label(frame, text="سوال خود را بپرسید:").grid(row=1, column=0, columnspan=2, sticky='w', padx=5, pady=5)
        self.query_text = tk.Text(frame, height=3, wrap=tk.WORD, font=("Segoe UI", 10))
        self.query_text.grid(row=2, column=0, columnspan=2, sticky='ew', padx=5, pady=5)

        self.search_button = ttk.Button(frame, text=" 🔍 جستجو ", command=self._start_search)
        self.search_button.grid(row=3, column=1, sticky='e', pady=5, padx=5)

    def _create_search_results_widgets(self):
        self.results_frame = ttk.LabelFrame(self, text="  📄 نتایج  ", padding="10")
        self.results_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self.results_frame.columnconfigure(0, weight=1)
        self.results_frame.rowconfigure(0, weight=1)

        self.placeholder_label = ttk.Label(self.results_frame, text="نتایج جستجو در اینجا نمایش داده خواهد شد.")
        self.placeholder_label.pack(pady=20)

    def populate_collections_dropdown(self):
        logger.info("Refreshing collections list...")
        collections = search_logic.list_available_collections()
        self.collection_dropdown['values'] = collections
        if collections:
            self.collection_dropdown.set(collections[0])
            logger.info(f"Found collections: {collections}")
        else:
            self.collection_dropdown.set("")
            logger.warning("No collections found.")

    def _start_search(self):
        query = self.query_text.get(1.0, tk.END).strip()
        collection = self.collection_var.get()

        if not query or not collection:
            messagebox.showerror("خطا", "لطفاً یک پایگاه داده را انتخاب کرده و سوال خود را وارد نمایید.")
            return

        self.search_button.config(state="disabled")
        self.placeholder_label.config(text="در حال جستجو...")

        thread = threading.Thread(target=self._search_worker, args=(collection, query))
        thread.daemon = True
        thread.start()

    def _search_worker(self, collection_name, query_text):
        results = search_logic.perform_search(collection_name, query_text)
        self.after(0, self._handle_search_completion, results)

    def _handle_search_completion(self, results):
        self.search_button.config(state="normal")
        self.display_results(results)

    def display_results(self, results: list):
        for widget in self.results_frame.winfo_children():
            widget.destroy()

        if not results:
            ttk.Label(self.results_frame, text="هیچ نتیجه‌ای یافت نشد.").pack(pady=20)
            return

        canvas = tk.Canvas(self.results_frame); scrollbar = ttk.Scrollbar(self.results_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw"); canvas.configure(yscrollcommand=scrollbar.set)

        for i, result in enumerate(results):
            self._create_result_widget(scrollable_frame, result, i + 1)

        canvas.pack(side="left", fill="both", expand=True); scrollbar.pack(side="right", fill="y")

    def _create_result_widget(self, parent, result: dict, rank: int):
        result_frame = ttk.Frame(parent, padding=10, borderwidth=1, relief="solid")
        result_frame.pack(fill="x", pady=5, padx=5, expand=True)

        header = f"🥇 نتیجه {rank} - امتیاز: {result.get('score', 0):.2f}"
        ttk.Label(result_frame, text=header, font=("Segoe UI", 11, "bold")).pack(anchor='w')

        source = result.get('source', 'نامشخص')
        ttk.Label(result_frame, text=f"📚 منبع: {source}", font=("Segoe UI", 9)).pack(anchor='w')

        text_preview = result.get('text', '')[:350] + "..."
        text_widget = tk.Text(result_frame, wrap=tk.WORD, height=4, font=("Segoe UI", 10), relief="flat", bg=result_frame.cget('bg'))
        text_widget.insert(tk.END, text_preview); text_widget.config(state="disabled")
        text_widget.pack(fill='x', pady=5)

        # Placeholder for detailed view
        ttk.Button(result_frame, text="[مشاهده کامل متن]").pack(anchor='w')

if __name__ == '__main__':
    root = tk.Tk()
    root.title("Test Searching Tab")
    root.geometry("800x700")

    import sys, os
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, project_root)

    tab = SearchingTab(root)
    tab.pack(expand=True, fill='both')

    # Dummy data for testing display
    dummy_results = [
        {'score': 0.89, 'source': 'کتاب_اکنکار_جلد1.docx » فصل 3', 'text': 'آگاهی روحانی به معنای درک عمیق از خود درونی است... این آگاهی از طریق تمرینات روزانه مراقبه و...'},
        {'score': 0.82, 'source': 'کتاب_اکنکار_جلد2.docx » فصل 1', 'text': 'راه دستیابی به آگاهی، گذر از ذهن محدود و...'}
    ]
    tab.display_results(dummy_results)

    root.mainloop()