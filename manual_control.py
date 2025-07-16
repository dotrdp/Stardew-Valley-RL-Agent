import tkinter as tk
import tkinter.font as tkFont
import re
from map_wrapper import Map
from player import player
from ENV import environment
from API import StardewModdingAPI

# ANSI to Tkinter color mapping (basic)
ANSI_COLORS = {
    '30': 'black',
    '31': 'red',
    '32': 'green',
    '33': 'yellow',
    '34': 'blue',
    '35': 'magenta',
    '36': 'cyan',
    '37': 'white',
    '90': 'gray',
    '91': 'red',
    '92': 'green',
    '93': 'yellow',
    '94': 'blue',
    '95': 'magenta',
    '96': 'cyan',
    '97': 'white',
}

def insert_ansi_text(text_widget, ansi_text):
    ansi_escape = re.compile(r'\x1b\[(\d+)m')
    pos = 0
    current_color = None
    for match in ansi_escape.finditer(ansi_text):
        start, end = match.span()
        code = match.group(1)
        # Insert text before the escape sequence
        if start > pos:
            segment = ansi_text[pos:start]
            if current_color:
                text_widget.insert(tk.END, segment, current_color)
            else:
                text_widget.insert(tk.END, segment)
        # Update color
        if code in ANSI_COLORS:
            current_color = ANSI_COLORS[code]
        elif code == '0':  # Reset
            current_color = None
        pos = end
    # Insert remaining text
    if pos < len(ansi_text):
        segment = ansi_text[pos:]
        if current_color:
            text_widget.insert(tk.END, segment, current_color)
        else:
            text_widget.insert(tk.END, segment)

def setup_tags(text_widget):
    for color in set(ANSI_COLORS.values()):
        text_widget.tag_config(color, foreground=color)

api = StardewModdingAPI(method="ssh+tty")
env = environment(api)
doll = player(env)
map = Map(api)

def update_map_display():
    map_display.delete("1.0", tk.END)
    insert_ansi_text(map_display, str(map.grid))

def handle_action(key):
    doll.normal_action(key, 200)
    update_map_display()

def handle_inventory(index):
    if 0 <= index < len(doll.inventory.items):
        doll.inventory.items[index]()
        update_map_display()

def show_map(event=None):
    update_map_display()

root = tk.Tk()
root.title("Manual Control")

map_display = tk.Text(root, font=("Consolas", 12))
map_display.pack(expand=True, fill='both')
setup_tags(map_display)

def update_font_size(event=None):
    # Get the current map grid as string
    map_str = str(map.grid)
    # Get widget dimensions
    widget_width = map_display.winfo_width()
    widget_height = map_display.winfo_height()
    # Start with a reasonable max font size
    font_size = 32
    family = "Consolas"
    while font_size >= 8:
        test_font = tkFont.Font(family=family, size=font_size)
        # Measure the longest line
        lines = map_str.splitlines()
        max_line_width = max((test_font.measure(line) for line in lines), default=0)
        total_height = test_font.metrics("linespace") * len(lines)
        # Check if fits
        if max_line_width <= widget_width and total_height <= widget_height:
            break
        font_size -= 1
    # Set the font
    map_display.configure(font=(family, font_size))
    update_map_display()

def on_resize(event):
    update_font_size()

root.bind('<Configure>', on_resize)

# Movement keys
root.bind('<w>', lambda e: handle_action("w"))
root.bind('<s>', lambda e: handle_action("s"))
root.bind('<a>', lambda e: handle_action("a"))
root.bind('<d>', lambda e: handle_action("d"))
root.bind('<c>', lambda e: handle_action("c"))

# Inventory keys
for i, key in enumerate(['1','2','3','4','5','6','7','8','9','0','q','e']):
    root.bind(f'<{key}>', lambda e, idx=i: handle_inventory(idx))

# Show map
root.bind('<m>', show_map)

update_map_display()
root.mainloop()
