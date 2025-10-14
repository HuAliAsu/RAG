import sys
import os
import tkinter as tk

# Add the project root to the Python path to allow imports from `src`
# This assumes that the script is run from the RAG_Workbench directory
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    from src.app import RAGWorkbenchApp
    from src.utils.logger import logger
except ImportError as e:
    # This provides a more user-friendly message if dependencies are not installed
    # or if the path is incorrect.
    print(f"FATAL: Could not import necessary modules: {e}")
    print("Please ensure you are running this from the 'RAG_Workbench' directory and have installed all requirements.")
    sys.exit(1)

def main():
    """
    Main function to initialize and run the RAG Workbench application.
    """
    try:
        logger.info("===================================================")
        logger.info("🚀 Starting RAG Workbench Application")
        logger.info("===================================================")

        root = tk.Tk()
        app = RAGWorkbenchApp(root)
        root.mainloop()

        logger.info("===================================================")
        logger.info("🛑 RAG Workbench Application Closed")
        logger.info("===================================================")

    except Exception as e:
        logger.critical(f"A fatal error occurred: {e}", exc_info=True)
        # Optionally show a message box to the user
        tk.messagebox.showerror("Fatal Error", f"An unexpected error occurred. Please check the logs.\n\n{e}")
    finally:
        # Ensure logs are written before exit
        logging.shutdown()


if __name__ == "__main__":
    # Add a basic check for the config file before starting
    if not os.path.exists('config.ini'):
        print("FATAL: config.ini not found. Please ensure the application is run from the root project directory.")
    else:
        # Need to import logging here to avoid circular dependencies with logger
        import logging
        import tkinter.messagebox
        main()