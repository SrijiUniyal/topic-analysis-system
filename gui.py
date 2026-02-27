import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
from src.predict import TopicPredictor


class TopicAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Topic Analysis System")
        self.root.geometry("900x780")
        self.root.configure(bg="#f4f6f9")

        self.style = ttk.Style()
        self.style.theme_use("clam")

        self.predictor = TopicPredictor()
        self.current_topic = ""

        self.setup_styles()
        self.setup_gui()

    # ---------------- CUSTOM STYLES ----------------
    def setup_styles(self):
        self.style.configure("Header.TLabel",
                             font=("Segoe UI", 20, "bold"),
                             background="#1e3d59",
                             foreground="white")

        self.style.configure("Card.TLabelframe",
                             background="white")

        self.style.configure("TButton",
                             font=("Segoe UI", 10, "bold"),
                             padding=6)

    # ---------------- GUI LAYOUT ----------------
    def setup_gui(self):

        # Header Bar
        header = ttk.Label(self.root,
                           text="  Topic Analysis System",
                           style="Header.TLabel",
                           anchor="w")
        header.pack(fill=tk.X)

        main_frame = tk.Frame(self.root, bg="#f4f6f9")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # ---------------- Topic Card ----------------
        topic_card = tk.LabelFrame(main_frame, text=" Topic Selection ",
                                   bg="white", font=("Segoe UI", 11, "bold"))
        topic_card.pack(fill=tk.X, pady=10)

        tk.Label(topic_card, text="Enter Topic:",
                 bg="white", font=("Segoe UI", 10)).pack(side=tk.LEFT, padx=10)

        self.topic_entry = tk.Entry(topic_card, width=35, font=("Segoe UI", 10))
        self.topic_entry.pack(side=tk.LEFT, padx=10, pady=10)

        tk.Button(topic_card, text="Set Topic",
                  bg="#1e3d59", fg="white",
                  command=self.set_topic).pack(side=tk.LEFT, padx=10)

        self.topic_label = tk.Label(main_frame,
                                    text="Current Topic: None",
                                    bg="#f4f6f9",
                                    font=("Segoe UI", 12, "bold"))
        self.topic_label.pack(pady=5)

        # ---------------- Input Card ----------------
        input_card = tk.LabelFrame(main_frame, text=" Your Input ",
                                   bg="white", font=("Segoe UI", 11, "bold"))
        input_card.pack(fill=tk.X, pady=10)

        self.input_text = scrolledtext.ScrolledText(
            input_card, height=4, font=("Segoe UI", 10))
        self.input_text.pack(fill=tk.X, padx=10, pady=5)
        self.input_text.bind("<KeyRelease>", self.update_word_count)

        self.word_count_label = tk.Label(
            input_card, text="Words: 0",
            bg="white", font=("Segoe UI", 9))
        self.word_count_label.pack(anchor="e", padx=10)

        tk.Label(input_card,
                 text="⚠ Minimum 2 words required for accurate prediction",
                 fg="red", bg="white",
                 font=("Segoe UI", 9)).pack(anchor="w", padx=10)

        tk.Button(input_card, text="Analyze",
                  bg="#28a745", fg="white",
                  font=("Segoe UI", 10, "bold"),
                  command=self.analyze_input).pack(pady=10)

        self.progress = ttk.Progressbar(input_card, mode="indeterminate")
        self.progress.pack(fill=tk.X, padx=10, pady=5)

        # ---------------- Results Card ----------------
        result_card = tk.LabelFrame(main_frame, text=" Analysis Results ",
                                    bg="white", font=("Segoe UI", 11, "bold"))
        result_card.pack(fill=tk.BOTH, expand=True, pady=10)

        self.sentiment_label = tk.Label(
            result_card,
            text="Sentiment: Not analyzed",
            bg="white",
            font=("Segoe UI", 14, "bold"))
        self.sentiment_label.pack(anchor="w", padx=10, pady=5)

        self.confidence_label = tk.Label(
            result_card,
            text="Confidence: --",
            bg="white",
            font=("Segoe UI", 11))
        self.confidence_label.pack(anchor="w", padx=10)

        self.confidence_bar = ttk.Progressbar(result_card, length=400)
        self.confidence_bar.pack(anchor="w", padx=10, pady=5)

        self.polarity_label = tk.Label(result_card,
                                       text="Polarity: --",
                                       bg="white")
        self.polarity_label.pack(anchor="w", padx=10)

        self.subjectivity_label = tk.Label(result_card,
                                           text="Subjectivity: --",
                                           bg="white")
        self.subjectivity_label.pack(anchor="w", padx=10)

        self.top_features_text = scrolledtext.ScrolledText(
            result_card, height=5, font=("Segoe UI", 10))
        self.top_features_text.pack(fill=tk.BOTH, padx=10, pady=5)

        self.response_text = scrolledtext.ScrolledText(
            result_card, height=6, font=("Segoe UI", 10))
        self.response_text.pack(fill=tk.BOTH, padx=10, pady=5)

    # ---------------- Word Counter ----------------
    def update_word_count(self, event=None):
        text = self.input_text.get(1.0, tk.END).strip()
        words = len(text.split())
        self.word_count_label.config(text=f"Words: {words}")

    # ---------------- Topic ----------------
    def set_topic(self):
        topic = self.topic_entry.get().strip()
        if topic:
            self.current_topic = topic
            self.topic_label.config(text=f"Current Topic: {topic}")
        else:
            messagebox.showwarning("Input Error", "Please enter a topic name")

    # ---------------- Analyze ----------------
    def analyze_input(self):
        if not self.current_topic:
            messagebox.showwarning("Topic Error", "Please set a topic first")
            return

        user_input = self.input_text.get(1.0, tk.END).strip()
        if len(user_input.split()) < 2:
            messagebox.showwarning(
                "Input Too Short",
                "Enter at least 2 words for accurate prediction."
            )
            return

        self.progress.start()
        thread = threading.Thread(
            target=self.perform_analysis, args=(user_input,))
        thread.daemon = True
        thread.start()

    # ---------------- Prediction ----------------
    def perform_analysis(self, user_input):
        result = self.predictor.predict_sentiment(
            self.current_topic, user_input)
        self.root.after(0, self.update_results, result)

    # ---------------- Update Results ----------------
    def update_results(self, result):
        self.progress.stop()

        sentiment = result["sentiment"].title()
        color = "#28a745" if sentiment == "Advantage" else "#dc3545"

        self.sentiment_label.config(
            text=f"Sentiment: {sentiment}", fg=color)

        confidence = result["confidence"]
        self.confidence_label.config(
            text=f"Confidence: {confidence:.2f}%")
        self.confidence_bar["value"] = confidence

        self.polarity_label.config(
            text=f"Polarity: {result['polarity']}")
        self.subjectivity_label.config(
            text=f"Subjectivity: {result['subjectivity']}")

        self.top_features_text.delete(1.0, tk.END)
        self.top_features_text.insert(
            tk.END,
            "Top Influential Words:\n\n" +
            "\n".join(result["top_features"])
        )

        self.response_text.delete(1.0, tk.END)
        self.response_text.insert(
            tk.END,
            "Opposite Perspective:\n\n" +
            "\n".join(result["opposite_responses"])
        )


def main():
    root = tk.Tk()
    app = TopicAnalysisGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()