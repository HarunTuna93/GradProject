"""
This program was created me: "Harun Tuna" as part of my graduation project.
Disclaimer: AI tools were used during the development process of this program.

Description:
"Book_Processor" assists with handling images (e.g., scanned book pages) in three modes:
1. Free Rename mode: Manually rename each image by typing a new name.
2. Pattern Rename mode: Use a pattern with '[]' placeholders to rename images in a structured fashion.
3. Page Separator mode: Split each image into two parts vertically. This can be used to reform double copies or twin column page structure.

Keybindings and Controls:
- Common Across All Modes:
  - Esc: Quits the application.
  - Tab: Skip the current image.
  - Enter: Confirm.
- In Page Separator Mode:
  - a/d: Move the top cutting line horizontally (left/right).
  - Left/Right Arrow Keys: Move the bottom cutting line horizontally.

Dependencies:
- Python 3.x
- tkinter
- Pillow

Usage:
1. Run `python ht_book_processor.py`.
2. A welcome popup prompts you to select a mode: Free Rename, Pattern Rename, or Page Separator.
3. Depending on the mode:
   - Free/Pattern Rename: Each image is displayed, and a popup appears for renaming or skipping. This is same for Pattern Rename mode.
   - Pattern Rename: Providing a pattern with '[]' placeholders for quick renaming e.g. "History Book Page []".
   - Page Separator: The image is displayed with adjustable lines. Use keys or drag with mouse and position the line to split.

Important Notes:
- Images are displayed scaled to fit the screen or 80% of it in separator mode. Scaling does not affect the actual output quality or the dimensions of the resulting split images.
- Renamed or split images are saved in the same directory as the original images with _part1 & _part2 suffix. Extension is same as original image.
"""

import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw


class ImageRenamer:
    # The main class.
    def __init__(self, master):
        self.master = master
        self.master.title("Book Processor")
        self.master.state('zoomed')  #Maximize the window for best viewing.
        self.mode = None
        self.pattern = None
        self.forbidden_chars = r'<>:"/\\|?*'  #These chars are not allowed in naming schema
        self.image_label = None
        self.canvas = None
        self.scale_top = None
        self.scale_bottom = None
        self.bottom_frame = None
        self.top_frame = None
        self.original_img = None
        self.display_img = None
        self.tk_img = None
        self.line_id = None
        self.top_line_x = 0
        self.bottom_line_x = 0
        self.separator_interface_created = False
        self.bind_escape_to_quit(self.master)
        self.welcome_popup()

    def bind_escape_to_quit(self, window):
        # Bind for quit.
        window.bind('<Escape>', lambda e: self.quit_app())

    def welcome_popup(self):
        # Welcome popup.
        popup = tk.Toplevel(self.master)
        popup.title("Welcome")
        self.bind_escape_to_quit(popup)
        # Center the popup window
        popup_width = 300
        popup_height = 200
        screen_w = popup.winfo_screenwidth()
        screen_h = popup.winfo_screenheight()
        x = (screen_w // 2) - (popup_width // 2)
        y = (screen_h // 2) - (popup_height // 2)
        popup.geometry(f"{popup_width}x{popup_height}+{x}+{y}")
        popup.transient(self.master)
        popup.grab_set()
        popup.focus_force()
        popup.lift()
        tk.Label(popup, text="Choose a Mode:").pack(pady=10)

        def choose_free():
            # Choice 1
            self.mode = "free"
            popup.destroy()
            self.init_directory_and_images()

        def choose_pattern():
            # Choice 2
            self.mode = "pattern"
            popup.destroy()
            self.init_directory_and_images()

        def choose_separator():
            # Choice 3
            self.mode = "separator"
            popup.destroy()
            self.init_directory_and_images()

        tk.Button(popup, text="Free Rename", command=choose_free).pack(pady=5)
        tk.Button(popup, text="Pattern Rename", command=choose_pattern).pack(pady=5)
        tk.Button(popup, text="Page Separator", command=choose_separator).pack(pady=5)

        self.master.wait_window(popup)

    def init_directory_and_images(self):
        # Prompt user for image directory
        self.directory = filedialog.askdirectory(title="Select Image Directory")
        if not self.directory:
            messagebox.showerror("Error", "No dir selected.")
            self.master.destroy()
            return

        valid_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')
        self.image_files = [f for f in os.listdir(self.directory) if f.lower().endswith(valid_extensions)]

        if not self.image_files:
            messagebox.showerror("Error", "No images found in the selected dir.")
            self.master.destroy()
            return

        self.current_index = 0
        # If in Pattern mode, ask for pattern
        if self.mode == "pattern":
            self.ask_for_pattern()
        self.show_image()

    def ask_for_pattern(self):
        popup = tk.Toplevel(self.master)
        popup.title("Enter Pattern")
        self.bind_escape_to_quit(popup)

        # Center the popup2
        popup_width = 500
        popup_height = 150
        screen_w = popup.winfo_screenwidth()
        screen_h = popup.winfo_screenheight()
        x = (screen_w // 2) - (popup_width // 2)
        y = (screen_h // 2) - (popup_height // 2)
        popup.geometry(f"{popup_width}x{popup_height}+{x}+{y}")
        popup.transient(self.master)
        popup.grab_set()
        popup.focus_force()
        popup.lift()
        tk.Label(popup, text="Enter a pattern with [].\nExample: 'Tarih Kitabı Sayfalar []-[]' or 'Kimya []'").pack(
            pady=10)
        pattern_entry = tk.Entry(popup, width=50)
        pattern_entry.pack(pady=5)
        pattern_entry.focus_set()

        def submit_pattern(event=None):
            p = pattern_entry.get().strip()
            if '[]' not in p:
                messagebox.showerror("Error", "Pattern must contain '[]'")
                return
            self.pattern = p
            popup.destroy()

        tk.Button(popup, text="OK", command=submit_pattern).pack(pady=5)
        pattern_entry.bind('<Return>', submit_pattern)
        self.master.wait_window(popup)

    def show_image(self):
        # Display current image or quit if done
        if self.current_index >= len(self.image_files):
            messagebox.showinfo("Done", "No more images.")
            self.master.quit()
            return

        image_path = os.path.join(self.directory, self.image_files[self.current_index])
        self.original_img = Image.open(image_path)

        screen_w = self.master.winfo_screenwidth()
        screen_h = self.master.winfo_screenheight()

        img_ratio = self.original_img.width / self.original_img.height
        screen_ratio = screen_w / screen_h

        if img_ratio > screen_ratio:
            new_w = screen_w
            new_h = int(new_w / img_ratio)
        else:
            new_h = screen_h
            new_w = int(new_h * img_ratio)

        # In separator mode, scale to 80%
        if self.mode == "separator":
            new_w = int(new_w * 0.8)
            new_h = int(new_h * 0.8)

        # Resize image for display if needed
        if self.original_img.width > new_w or self.original_img.height > new_h:
            self.display_img = self.original_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        else:
            self.display_img = self.original_img.copy()

        self.tk_img = ImageTk.PhotoImage(self.display_img)
        self.master.title(f"Book Processor - {self.image_files[self.current_index]}")

        # Mode UI
        if self.mode == "free":
            self.show_mode_free()
        elif self.mode == "pattern":
            self.show_mode_pattern()
        else:
            self.show_mode_separator()

    def show_mode_free(self):
        # Free Rename mode
        if self.canvas:
            self.canvas.destroy()
            self.canvas = None
        if self.top_frame:
            self.top_frame.pack_forget()
        if self.bottom_frame:
            self.bottom_frame.pack_forget()

        if not self.image_label:
            self.image_label = tk.Label(self.master, bg="black")
            self.image_label.pack(expand=True, fill=tk.BOTH)

        self.image_label.config(image=self.tk_img)
        self.show_rename_popup_free()

    def show_mode_pattern(self):
        # Pattern Rename mode
        if self.canvas:
            self.canvas.destroy()
            self.canvas = None
        if self.top_frame:
            self.top_frame.pack_forget()
        if self.bottom_frame:
            self.bottom_frame.pack_forget()

        if not self.image_label:
            self.image_label = tk.Label(self.master, bg="black")
            self.image_label.pack(expand=True, fill=tk.BOTH)

        self.image_label.config(image=self.tk_img)
        self.show_rename_popup_pattern()

    def show_mode_separator(self):
        # Page Separator mode
        if self.image_label:
            self.image_label.destroy()
            self.image_label = None

        if not self.separator_interface_created:
            self.top_frame = tk.Frame(self.master)
            self.top_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
            self.bottom_frame = tk.Frame(self.master)
            self.bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
            self.create_separator_controls()
            self.separator_interface_created = True
        else:
            self.top_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
            self.bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)

        if self.canvas:
            self.canvas.destroy()

        self.canvas = tk.Canvas(self.master, bg="black", highlightthickness=0,
                                width=self.display_img.width, height=self.display_img.height)
        self.canvas.pack(expand=True, fill=tk.BOTH)

        self.canvas_img = self.canvas.create_image(
            self.display_img.width // 2, self.display_img.height // 2, image=self.tk_img, anchor='center'
        )
        self.canvas.config(scrollregion=(0, 0, self.display_img.width, self.display_img.height))

        self.top_line_x = self.display_img.width // 2
        self.bottom_line_x = self.display_img.width // 2
        self.draw_line()

        self.scale_top.config(to=self.display_img.width)
        self.scale_bottom.config(to=self.display_img.width)
        self.scale_top.set(self.top_line_x)
        self.scale_bottom.set(self.bottom_line_x)

        self.master.bind('<Tab>', lambda e: self.skip())
        # 'a'/'d' for top line
        self.master.bind('<a>', lambda e: self.move_slider(self.scale_top, -1))
        self.master.bind('<d>', lambda e: self.move_slider(self.scale_top, 1))
        # Left/Right for bottom line
        self.master.bind('<Left>', lambda e: self.move_slider(self.scale_bottom, -1))
        self.master.bind('<Right>', lambda e: self.move_slider(self.scale_bottom, 1))

    def create_separator_controls(self):
        #Create sliders and controls in third mode.
        self.scale_top = tk.Scale(
            self.top_frame, from_=0, to=100, orient=tk.HORIZONTAL,
            command=self.update_line_position_top, label="Top Line X"
        )
        self.scale_top.set(self.top_line_x)
        self.scale_top.pack(fill=tk.X)

        self.scale_bottom = tk.Scale(
            self.bottom_frame, from_=0, to=100, orient=tk.HORIZONTAL,
            command=self.update_line_position_bottom, label="Bottom Line X"
        )
        self.scale_bottom.set(self.bottom_line_x)
        self.scale_bottom.pack(fill=tk.X)

        info_frame = tk.Frame(self.bottom_frame)
        info_frame.pack(pady=5)
        tk.Label(info_frame, text="Press 'Enter' to split, 'Tab' to skip, 'Esc' to quit.").pack()

        button_frame = tk.Frame(self.bottom_frame)
        button_frame.pack(pady=5)

        skip_btn = tk.Button(button_frame, text="Skip (Tab)", command=self.skip)
        skip_btn.pack(side=tk.LEFT, padx=5)

        quit_btn = tk.Button(button_frame, text="Quit (Esc)", command=self.quit_app)
        quit_btn.pack(side=tk.LEFT, padx=5)

        small_label = tk.Label(info_frame, text="a/d: top slider | ←/→: bottom slider",
                               font=('TkDefaultFont', 8))
        small_label.pack(side=tk.RIGHT, padx=10)

        self.master.bind('<Return>', self.split_image)

    def move_slider(self, scale, delta):
        #Move slider with limiting.
        val = scale.get() + delta
        width = self.display_img.width
        if val < 0:
            val = 0
        if val > width:
            val = width
        scale.set(val)

    def draw_line(self):
        #Draws the cut line.
        if self.line_id:
            self.canvas.delete(self.line_id)
        self.line_id = self.canvas.create_line(
            self.top_line_x, 0,
            self.bottom_line_x, self.display_img.height,
            fill="red", width=2
        )

    def show_rename_popup_free(self):
        #Rename popup for Free mode.
        popup = tk.Toplevel(self.master)
        popup.title("Free Rename")
        self.bind_escape_to_quit(popup)

        # Center the popup
        popup_width = 400
        popup_height = 200
        screen_w = popup.winfo_screenwidth()
        screen_h = popup.winfo_screenheight()
        x = (screen_w // 2) - (popup_width // 2)
        y = (screen_h // 2) - (popup_height // 2)
        popup.geometry(f"{popup_width}x{popup_height}+{x}+{y}")

        popup.transient(self.master)
        popup.grab_set()
        popup.focus_force()
        popup.lift()

        current_name = self.image_files[self.current_index]
        total_images = len(self.image_files)
        current_num = self.current_index + 1

        tk.Label(popup, text=f"Current name: {current_name}  ({current_num}/{total_images})").pack(pady=(10, 5))
        tk.Label(popup, text="Enter new name (without extension):").pack(pady=5)

        name_entry = tk.Entry(popup, width=40)
        name_entry.pack(pady=5)
        name_entry.focus_set()

        def delayed_focus():
            popup.focus_force()
            popup.lift()
            name_entry.focus_set()

        popup.after(100, delayed_focus)

        def rename_and_next(event=None):
            # Confirm the entered name and proceed
            new_name = name_entry.get().strip()
            if new_name:
                # Check chars
                if any(ch in new_name for ch in self.forbidden_chars):
                    messagebox.showerror("Error", f"Filename contains forbidden chars: {self.forbidden_chars}")
                    return
                old_path = os.path.join(self.directory, self.image_files[self.current_index])
                root, ext = os.path.splitext(self.image_files[self.current_index])
                new_path = os.path.join(self.directory, new_name + ext)

                if os.path.exists(new_path):
                    messagebox.showerror("Error", f"File '{new_path}' already exists.")
                    return
                # Rename on disk
                os.rename(old_path, new_path)
                self.image_files[self.current_index] = new_name + ext

            self.current_index += 1
            popup.destroy()
            self.show_image()

        def quit_app():
            popup.destroy()
            self.quit_app()

        button_frame = tk.Frame(popup)
        button_frame.pack(pady=10)

        rename_btn = tk.Button(button_frame, text="Rename & Next (Enter)", command=rename_and_next)
        rename_btn.pack(side=tk.LEFT, padx=5)

        # 'Skip' now calls self.skip()
        skip_btn = tk.Button(button_frame, text="Skip (Tab)", command=self.skip)
        skip_btn.pack(side=tk.LEFT, padx=5)

        quit_btn = tk.Button(button_frame, text="Quit (Esc)", command=quit_app)
        quit_btn.pack(side=tk.LEFT, padx=5)

        name_entry.bind('<Return>', rename_and_next)
        popup.bind('<Tab>', lambda e: self.skip())

        self.master.wait_window(popup)

    def show_rename_popup_pattern(self):
        #Pattern rename popup.
        popup = tk.Toplevel(self.master)
        popup.title("Pattern Rename")
        self.bind_escape_to_quit(popup)

        popup_width = 500
        popup_height = 300
        screen_w = popup.winfo_screenwidth()
        screen_h = popup.winfo_screenheight()
        x = (screen_w // 2) - (popup_width // 2)
        y = (screen_h // 2) - (popup_height // 2)
        popup.geometry(f"{popup_width}x{popup_height}+{x}+{y}")

        popup.transient(self.master)
        popup.grab_set()
        popup.focus_force()
        popup.lift()

        current_name = self.image_files[self.current_index]
        total_images = len(self.image_files)
        current_num = self.current_index + 1

        tk.Label(popup, text=f"Current name: {current_name}  ({current_num}/{total_images})").pack(pady=(10, 5))
        parts = self.pattern.split('[]')
        placeholders_count = len(parts) - 1

        pattern_display_frame = tk.Frame(popup)
        pattern_display_frame.pack(pady=10)
        entries = []

        for i in range(placeholders_count):
            if parts[i]:
                tk.Label(pattern_display_frame, text=parts[i]).pack(side=tk.LEFT)
            e = tk.Entry(pattern_display_frame, width=10)
            e.pack(side=tk.LEFT)
            entries.append(e)
        if parts[-1]:
            tk.Label(pattern_display_frame, text=parts[-1]).pack(side=tk.LEFT)

        def delayed_focus():
            popup.focus_force()
            popup.lift()
            if entries:
                entries[0].focus_set()

        popup.after(100, delayed_focus)

        def rename_and_next():
            # Construct final filename
            filled_parts = []
            for e in entries:
                val = e.get().strip()
                if any(ch in val for ch in self.forbidden_chars):
                    messagebox.showerror("Error", f"Filename contains forbidden chars: {self.forbidden_chars}")
                    return
                filled_parts.append(val)

            new_name = ""
            for i in range(placeholders_count):
                new_name += parts[i] + filled_parts[i]
            new_name += parts[-1]

            new_name = new_name.strip()
            if not new_name:
                self.current_index += 1
                popup.destroy()
                self.show_image()
                return

            old_path = os.path.join(self.directory, self.image_files[self.current_index])
            root, ext = os.path.splitext(self.image_files[self.current_index])
            new_path = os.path.join(self.directory, new_name + ext)

            if os.path.exists(new_path):
                messagebox.showerror("Error", f"File '{new_path}' already exists.")
                return

            os.rename(old_path, new_path)
            self.image_files[self.current_index] = new_name + ext

            self.current_index += 1
            popup.destroy()
            self.show_image()

        def quit_app():
            popup.destroy()
            self.quit_app()

        button_frame = tk.Frame(popup)
        button_frame.pack(pady=10)

        rename_btn = tk.Button(button_frame, text="Rename & Next (Enter)", command=rename_and_next)
        rename_btn.pack(side=tk.LEFT, padx=5)

        skip_btn = tk.Button(button_frame, text="Skip (Tab)", command=self.skip)
        skip_btn.pack(side=tk.LEFT, padx=5)

        quit_btn = tk.Button(button_frame, text="Quit (Esc)", command=quit_app)
        quit_btn.pack(side=tk.LEFT, padx=5)

        def on_entry_return(i, event=None):
            # Move to next placeholder or confirm if last
            if i < len(entries) - 1:
                entries[i + 1].focus_set()
            else:
                rename_and_next()

        for i, e in enumerate(entries):
            e.bind('<Return>', lambda event, idx=i: on_entry_return(idx, event))

        popup.bind('<Tab>', lambda e: self.skip())
        self.master.wait_window(popup)

    def update_line_position_top(self, val):
        #Top line updater
        val = int(float(val))
        self.top_line_x = val
        self.draw_line()

    def update_line_position_bottom(self, val):
        #Bottom line updater
        val = int(float(val))
        self.bottom_line_x = val
        self.draw_line()

    def split_image(self, event=None):
        #Image split function
        scale_x = self.original_img.width / self.display_img.width
        top_x_original = int(self.top_line_x * scale_x)
        bottom_x_original = int(self.bottom_line_x * scale_x)
        width = self.original_img.width
        height = self.original_img.height

        left_mask = Image.new('L', (width, height), 0)
        right_mask = Image.new('L', (width, height), 0)
        draw_left = ImageDraw.Draw(left_mask)
        draw_right = ImageDraw.Draw(right_mask)

        left_polygon = [(0, 0), (top_x_original, 0), (bottom_x_original, height), (0, height)]
        right_polygon = [(top_x_original, 0), (width, 0), (width, height), (bottom_x_original, height)]

        draw_left.polygon(left_polygon, fill=255)
        draw_right.polygon(right_polygon, fill=255)

        left_part = Image.new('RGBA', (width, height))
        right_part = Image.new('RGBA', (width, height))

        left_part.paste(self.original_img, (0, 0), left_mask)
        right_part.paste(self.original_img, (0, 0), right_mask)

        left_part = left_part.convert('RGB')
        right_part = right_part.convert('RGB')

        current_name = self.image_files[self.current_index]
        root, ext = os.path.splitext(current_name)
        left_name = f"{root}_part1{ext}"
        right_name = f"{root}_part2{ext}"

        left_path = os.path.join(self.directory, left_name)
        right_path = os.path.join(self.directory, right_name)

        left_part.save(left_path)
        right_part.save(right_path)
        self.current_index += 1
        self.show_image()

    def skip(self):
        #Skip function.
        self.current_index += 1
        self.show_image()

    def quit_app(self):
        #Quit Function
        self.master.quit()
        self.master.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageRenamer(root)
    root.mainloop()
