import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

from src.utils.logger import logger
from src.features.indexing.indexing_ui import IndexingTab
from src.features.searching.searching_ui import SearchingTab

class RAGWorkbenchApp:
    """
    The main application window for the RAG Workbench.
    It sets up the main window, theme, and tabs.
    """
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("میز کار هوشمند RAG برای اسناد فارسی (RAG Workbench for Persian Documents)")
        self.root.geometry("1000x800")

        # Apply a theme
        self.style = ttk.Style(self.root)
        self.style.theme_use('clam')

        self._setup_ui()
        logger.info("RAGWorkbenchApp UI initialized successfully.")

    def _setup_ui(self):
        """
        Creates the main UI components, including the tabbed notebook.
        """
        # Main container frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(expand=True, fill='both')

        # Create the Tabbed Notebook
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(expand=True, fill='both')

        # --- Create and Add Tabs ---
        self._add_indexing_tab()
        self._add_searching_tab()

        # Add a status bar
        self.status_bar = ttk.Label(self.root, text="  آماده  |  Ollama: ❔  |  ChromaDB: ❔  ", relief=tk.SUNKEN, anchor='w')
        self.status_bar.pack(side='bottom', fill='x')

    def _add_indexing_tab(self):
        """Initializes and adds the IndexingTab to the notebook."""
        indexing_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(indexing_frame, text='  ایندکسینگ (Indexing)  ')
        self.indexing_tab = IndexingTab(indexing_frame)
        self.indexing_tab.pack(expand=True, fill='both')
        logger.info("Indexing tab loaded.")

    def _add_searching_tab(self):
        """Initializes and adds the SearchingTab to the notebook."""
        searching_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(searching_frame, text='  جستجو (Searching)  ')
        self.searching_tab = SearchingTab(searching_frame)
        self.searching_tab.pack(expand=True, fill='both')
        logger.info("Searching tab loaded.")

if __name__ == '__main__':
    # This allows running this file directly for quick UI testing.
    # Note: For full functionality, run from main.py
    root = tk.Tk()
    app = RAGWorkbenchApp(root)
    root.mainloop()