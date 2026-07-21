"""
Generate three fake book spines/covers on a single A4 landscape PDF.

Physical dimensions of each fake cover:
    Height      = 18 cm
    Left panel  = 3 cm
    Spine       = w cm, where 2 <= w <= 3 cm
    Right panel = 3 cm

Total width = 6 + w cm

The three covers are placed side-by-side on an A4 landscape sheet.

Requires:
    pip install reportlab
"""

import random

from ...PythonicUtilities.path_tools import get_parent_path_of_file, take_outFileNameTermination

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.colors import Color
from reportlab.pdfbase.pdfmetrics import stringWidth, getFont

import fitz


# ============================================================================
# USER CONFIGURATION
# ============================================================================

# Book information
TITLE = ["INTRODUCTORY REAL ANALYSIS", "TENSOR CALCULUS", "VECTOR AND TENSOR ANALYSIS"]

AUTHORS = ["Kolmogorov\n& Fomin", "Synge\n& Schild", "G. E. Hay"]

BOOK_NUMBER = ["0-486-61226-0", "0-486-63612-7", "0-486-60109-9"]

BOOK_PUBLISHER = ["Dover", "Dover", "Dover"]


# Cover and font colors as RGB values in the range [0, 255]
COVER_RGB = [(30, 135, 105), (52, 68, 174), (122, 55, 210)]

FONT_RGB = [(245, 205, 115), (245, 205, 115), (245, 205, 115)]


# --------------------------------------------------------------------------
# Spine width
#
# Set to None to generate a random width between 2 and 3 cm.
#
# Or explicitly choose a value, for example:
#
# SPINE_WIDTH_CM = 2.4
# --------------------------------------------------------------------------

SPINE_WIDTH_CM = None


# Output file
OUTPUT_FILE = get_parent_path_of_file()+"//fake_book_spines.png"


# ============================================================================
# PAGE SETTINGS
# ============================================================================

PAGE_WIDTH, PAGE_HEIGHT = landscape(A4)


# A4 dimensions in centimeters:
#
# Landscape:
#     width  = 29.7 cm
#     height = 21.0 cm
#
# The fake cover height is 18 cm.
#
# Therefore, vertically:
#
#     top margin    = (21 - 18) / 2 = 1.5 cm
#
# horizontally, the three covers are centered.


# ============================================================================
# DIMENSIONS
# ============================================================================

COVER_HEIGHT_CM = 18.0

LEFT_PANEL_WIDTH_CM = 3.0
RIGHT_PANEL_WIDTH_CM = 3.0

MIN_SPINE_WIDTH_CM = 2.0
MAX_SPINE_WIDTH_CM = 3.0


# ============================================================================
# TYPOGRAPHY
# ============================================================================

# Font used for the main title
TITLE_FONT = "Helvetica-Bold"

# Font used for authors
AUTHOR_FONT = "Helvetica-Bold"

# Font used for book number/publisher information
BOOK_NUMBER_FONT = "Helvetica"

BOOK_PUBLISHER_FONT = "Helvetica-Bold"


# Font sizes in points
#
# These are deliberately chosen to resemble the photographed Dover spine.
TITLE_FONT_SIZE = 0.6
AUTHOR_FONT_SIZE = 0.5
BOOK_NUMBER_FONT_SIZE = 0.2
BOOK_PUBLISHER_FONT_SIZE = 0.3

# Selects the ratios of the cover height to position the beginning of each
# text excerpt

AUTHORS_COVER_RATIO = 0.1
TITLE_COVER_RATIO = 0.5
PUBLISHER_COVER_RATIO = 0.85
NUMBER_COVER_RATIO = 0.95

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def cm_to_pt(value_cm):
    """
    Converts centimeters to PDF points.
    """
    return int(round(value_cm * 72.0 / 2.54))


def rgb_to_reportlab(rgb):
    """
    Converts an RGB tuple in [0, 255] to a ReportLab Color.
    """
    return Color(
        rgb[0] / 255.0,
        rgb[1] / 255.0,
        rgb[2] / 255.0
    )


def fit_text_to_width(
    text,
    font_name,
    initial_font_size,
    maximum_width,
    minimum_font_size=4
):
    """
    Reduces the font size until the text fits within maximum_width.
    """

    font = getFont(font_name)

    # ReportLab font metrics are expressed in units of 1/1000
    # of the font size.
    ascent = font.face.ascent / 1000.0
    descent = abs(font.face.descent) / 1000.0

    # Total font height relative to the font size.
    font_height_factor = ascent + descent

    # Actual height at the requested font size.
    actual_font_height = (
        initial_font_size * font_height_factor
    )

    # If the font already fits, keep the requested font size.
    if actual_font_height <= maximum_width:

        font_size = initial_font_size

    # Otherwise, reduce the font size so that the font height is
    # exactly the maximum available height.
    else:

        font_size = (
            maximum_width /
            font_height_factor
        )

        # Do not go below the minimum allowed font size.
        font_size = max(
            font_size,
            minimum_font_size
        )

    print("The font size from '"+str(text)+"' is "+str(initial_font_size)+" and was corrected"+
    " to width of "+str(font_size)+"; the maximum width is "+str(maximum_width)+"\n")

    return font_size


def draw_rotated_centered_text(
    pdf,
    text,
    x,
    y,
    font_name,
    font_size,
    color
):
    """
    Draws horizontally oriented text centered around (x, y).

    This function is useful for the small horizontal labels.
    """

    pdf.setFont(font_name, font_size)
    pdf.setFillColor(color)

    text_width = stringWidth(
        text,
        font_name,
        font_size
    )

    pdf.drawString(
        x - text_width / 2.0,
        y,
        text
    )


def draw_vertical_centered_text(
    pdf,
    text,
    x,
    y,
    font_name,
    font_size,
    color,
    max_width=None,
    line_spacing=None
):
    """
    Draws text rotated by 90 degrees.

    The text is centered vertically along the spine.

    Multiple lines can be provided using '\\n'. For example:

        "Kolmogorov\\n& Fomin"

    The complete multiline text block is centered around (x, y).

    The original spine design is based on the physical book:

        Kolmogorov
        & Fomin

        INTRODUCTORY REAL ANALYSIS

        Dover 0-486-61226-0

    The title is rotated and runs along the long axis of the spine.
    """

    # If no line spacing is provided, use the font size as the
    # distance between consecutive baselines

    if line_spacing is None:

        line_spacing = font_size

    # Adjust the font size if a maximum width is specified

    if max_width is not None:

        font_size = fit_text_to_width(
            text,
            font_name,
            font_size,
            max_width
        )

    # Splits the text into individual lines

    text_lines = text.split("\n")

    # Calculates the total height occupied by the multiline text

    total_height = (
        (len(text_lines) - 1) * line_spacing
    )

    pdf.saveState()

    pdf.setFillColor(color)

    pdf.setFont(
        font_name,
        font_size
    )

    pdf.translate(
        x,
        y
    )

    pdf.rotate(90)

    # Draws each line such that the complete block is vertically
    # centered around the origin

    for line_index, line in enumerate(text_lines):

        # Calculates the vertical position of the current line
        # relative to the center of the multiline block

        line_y = (
            total_height / 2.0
            - line_index * line_spacing
        )

        # Calculates the horizontal width of the current line

        # Calculates the width of every line
        line_widths = [
            stringWidth(line, font_name, font_size)
            for line in text_lines
        ]

        # Finds the width of the longest line
        maximum_line_width = max(line_widths)

        # Draws each line
        for line_index, line in enumerate(text_lines):

            line_y = (
                total_height / 2.0
                - line_index * line_spacing
            )

            # All lines start at the same horizontal position
            pdf.drawString(
                -maximum_line_width / 2.0,
                line_y,
                line
            )

    pdf.restoreState()


def draw_arrow(
    pdf,
    x,
    y,
    direction,
    color
):
    """
    Draws a small arrow indicating where the ruler should be placed
    to fold the fake cover.

    direction:
        "down"  -> arrow points downward
        "up"    -> arrow points upward
    """

    pdf.saveState()

    pdf.setStrokeColor(color)
    pdf.setFillColor(color)

    arrow_length = cm_to_pt(0.35)
    arrow_head = cm_to_pt(0.12)

    if direction == "down":

        # Shaft
        pdf.line(
            x,
            y + arrow_length,
            x,
            y
        )

        # Arrowhead
        pdf.line(
            x,
            y,
            x - arrow_head,
            y + arrow_head
        )

        pdf.line(
            x,
            y,
            x + arrow_head,
            y + arrow_head
        )

    elif direction == "up":

        # Shaft
        pdf.line(
            x,
            y - arrow_length,
            x,
            y
        )

        # Arrowhead
        pdf.line(
            x,
            y,
            x - arrow_head,
            y - arrow_head
        )

        pdf.line(
            x,
            y,
            x + arrow_head,
            y - arrow_head
        )

    pdf.restoreState()


# ============================================================================
# DRAW ONE FAKE BOOK COVER
# ============================================================================

def draw_fake_book_cover(
    pdf,
    x,
    y,
    spine_width_cm, cover_index
):
    """
    Draws one fake book cover.

    Geometry:

        3 cm       w cm       3 cm
    |----------|----------|----------|
    |          |          |          |
    |          |          |          |
    |          |   SPINE  |          |
    |          |          |          |
    |          |          |          |
    |----------|----------|----------|

    The dividing lines between panels are deliberately NOT drawn.

    Only arrows at the top and bottom indicate the fold locations.
    """

    cover_height = cm_to_pt(
        COVER_HEIGHT_CM
    )

    left_width = cm_to_pt(
        LEFT_PANEL_WIDTH_CM
    )

    spine_width = cm_to_pt(
        spine_width_cm
    )

    right_width = cm_to_pt(
        RIGHT_PANEL_WIDTH_CM
    )

    total_width = (
        left_width
        + spine_width
        + right_width
    )

    cover_color = rgb_to_reportlab(
        COVER_RGB[cover_index]
    )

    font_color = rgb_to_reportlab(
        FONT_RGB[cover_index]
    )


    # ------------------------------------------------------------------------
    # Draw the complete outer rectangle
    # ------------------------------------------------------------------------

    pdf.setFillColor(
        cover_color
    )

    pdf.setStrokeColor(
        font_color
    )

    pdf.setLineWidth(
        0.6
    )

    pdf.rect(
        x,
        y,
        total_width,
        cover_height,
        stroke=1,
        fill=1
    )


    # ------------------------------------------------------------------------
    # Determine spine center
    # ------------------------------------------------------------------------

    spine_x_left = (
        x
        + left_width
    )

    spine_x_right = (
        spine_x_left
        + spine_width
    )

    spine_center_x = (
        spine_x_left
        + spine_width / 2.0
    )


    # ------------------------------------------------------------------------
    # Text positioning
    # ------------------------------------------------------------------------

    # The main title is placed vertically along the spine.

    draw_vertical_centered_text(
        pdf=pdf,
        text=TITLE[cover_index],
        x=spine_center_x,
        y=y + cover_height * TITLE_COVER_RATIO,
        font_name=TITLE_FONT,
        font_size=cm_to_pt(TITLE_FONT_SIZE),
        color=font_color,
        max_width=int(round(spine_width * 0.85))
    )


    # ------------------------------------------------------------------------
    # Authors
    #
    # The authors are placed near the top of the spine.
    # ------------------------------------------------------------------------

    authors_font_size = fit_text_to_width(
        AUTHORS[cover_index],
        AUTHOR_FONT,
        cm_to_pt(AUTHOR_FONT_SIZE),
        spine_width * 0.85
    )

    draw_vertical_centered_text(
        pdf=pdf,
        text=AUTHORS[cover_index],
        x=spine_center_x,
        y=y + cover_height * AUTHORS_COVER_RATIO,
        font_name=AUTHOR_FONT,
        font_size=authors_font_size,
        color=font_color,
        max_width=int(round(spine_width * 0.85))
    )


    # ------------------------------------------------------------------------
    # Book number / publisher
    #
    # Positioned near the bottom of the spine.
    # ------------------------------------------------------------------------

    book_number_font_size = fit_text_to_width(
        BOOK_NUMBER[cover_index],
        BOOK_NUMBER_FONT,
        cm_to_pt(BOOK_NUMBER_FONT_SIZE),
        spine_width * 0.85
    )

    book_publisher_font_size = fit_text_to_width(
        BOOK_PUBLISHER[cover_index],
        BOOK_PUBLISHER_FONT,
        cm_to_pt(BOOK_PUBLISHER_FONT_SIZE),
        spine_width * 0.85
    )

    draw_vertical_centered_text(
        pdf=pdf,
        text=BOOK_NUMBER[cover_index],
        x=spine_center_x,
        y=y + cover_height * NUMBER_COVER_RATIO,
        font_name=BOOK_NUMBER_FONT,
        font_size=book_number_font_size,
        color=font_color,
        max_width=int(round(spine_width * 0.85))
    )

    draw_vertical_centered_text(
        pdf=pdf,
        text=BOOK_PUBLISHER[cover_index],
        x=spine_center_x,
        y=y + cover_height * PUBLISHER_COVER_RATIO,
        font_name=BOOK_PUBLISHER_FONT,
        font_size=book_publisher_font_size,
        color=font_color,
        max_width=int(round(spine_width * 0.85))
    )


    # ------------------------------------------------------------------------
    # Fold indicators
    #
    # There are NO vertical dividing lines.
    #
    # The arrows indicate the two fold locations:
    #
    #       ↓       ↓
    #       |       |
    #       | spine |
    #
    #       ↑       ↑
    #
    # The user places a ruler between each pair of arrows and folds.
    # ------------------------------------------------------------------------

    arrow_offset = cm_to_pt(
        0.35
    )

    arrow_y_top = (
        y
        + cover_height
        + cm_to_pt(0.25)
    )

    arrow_y_bottom = (
        y
        - cm_to_pt(0.25)
    )


    # Top arrows

    draw_arrow(
        pdf,
        spine_x_left,
        arrow_y_top,
        "down",
        font_color
    )

    draw_arrow(
        pdf,
        spine_x_right,
        arrow_y_top,
        "down",
        font_color
    )


    # Bottom arrows

    draw_arrow(
        pdf,
        spine_x_left,
        arrow_y_bottom,
        "up",
        font_color
    )

    draw_arrow(
        pdf,
        spine_x_right,
        arrow_y_bottom,
        "up",
        font_color
    )


# ============================================================================
# CREATE PDF
# ============================================================================

def create_pdf():

    # ------------------------------------------------------------------------
    # Select spine width
    # ------------------------------------------------------------------------

    if SPINE_WIDTH_CM is None:

        spine_width_cm = random.uniform(
            MIN_SPINE_WIDTH_CM,
            MAX_SPINE_WIDTH_CM
        )

    else:

        spine_width_cm = SPINE_WIDTH_CM

        if not (
            MIN_SPINE_WIDTH_CM
            <= spine_width_cm
            <= MAX_SPINE_WIDTH_CM
        ):
            raise ValueError(
                "SPINE_WIDTH_CM must be between "
                "2 and 3 cm."
            )


    # ------------------------------------------------------------------------
    # Calculate dimensions
    # ------------------------------------------------------------------------

    cover_width_cm = (
        LEFT_PANEL_WIDTH_CM
        + spine_width_cm
        + RIGHT_PANEL_WIDTH_CM
    )

    cover_width = cm_to_pt(
        cover_width_cm
    )

    cover_height = cm_to_pt(
        COVER_HEIGHT_CM
    )


    # ------------------------------------------------------------------------
    # Check whether three covers fit
    # ------------------------------------------------------------------------

    total_covers_width = (
        3.0 * cover_width
    )

    if total_covers_width > PAGE_WIDTH:

        raise ValueError(
            "The three covers do not fit on an A4 "
            "landscape page."
        )


    # ------------------------------------------------------------------------
    # Create PDF
    # ------------------------------------------------------------------------

    pdf = canvas.Canvas(
        OUTPUT_FILE,
        pagesize=landscape(A4)
    )


    # ------------------------------------------------------------------------
    # Calculate margins
    # ------------------------------------------------------------------------

    horizontal_gap = (
        PAGE_WIDTH
        - total_covers_width
    ) / 4.0


    # Vertically center the 18 cm cover on the 21 cm A4 height.

    vertical_margin = (
        PAGE_HEIGHT
        - cover_height
    ) / 2.0


    # ------------------------------------------------------------------------
    # Draw the three covers
    # ------------------------------------------------------------------------

    for cover_index in range(3):

        x = (
            horizontal_gap
            + cover_index
            * (
                cover_width
                + horizontal_gap
            )
        )

        draw_fake_book_cover(
            pdf=pdf,
            x=x,
            y=vertical_margin,
            spine_width_cm=spine_width_cm, cover_index=cover_index
        )


    # ------------------------------------------------------------------------
    # Add a small technical note outside the covers
    #
    # This is intentionally omitted from the printable region because
    # the goal is to make the sheet directly usable.
    # ------------------------------------------------------------------------


    # Finish PDF

    pdf.showPage()

    pdf.save()

    pdf_to_png(
    OUTPUT_FILE,
    take_outFileNameTermination(OUTPUT_FILE)+".png",
    dpi=300
    )


    print(
        "PDF successfully generated:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        f"Spine width: {spine_width_cm:.3f} cm"
    )

    print(
        f"Total cover width: {cover_width_cm:.3f} cm"
    )

    print(
        f"Cover height: {COVER_HEIGHT_CM:.3f} cm"
    )


def pdf_to_png(pdf_path, png_path, page_number=0, dpi=300):

    # Opens the PDF
    pdf_document = fitz.open(pdf_path)

    # Gets the desired page
    page = pdf_document.load_page(page_number)

    # Converts DPI to a scaling factor
    zoom = dpi / 72.0

    matrix = fitz.Matrix(zoom, zoom)

    # Renders the page
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)

    # Saves as PNG
    pixmap.save(png_path)

    pdf_document.close()

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":

    create_pdf()