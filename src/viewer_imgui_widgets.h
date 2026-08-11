#pragma once

namespace viewer::imgui_widgets
{
bool checkbox(const char* label, bool* value);
bool radio_button(const char* label, int* value, int option);
bool slider_float(const char* label,
                  float* value,
                  float min_value,
                  float max_value,
                  const char* display_format = "%.3f",
                  float power = 1.0f);
bool slider_int(const char* label, int* value, int min_value, int max_value, const char* display_format = "%.0f");
bool button(const char* label, float width);
bool button_full_width(const char* label);
void section_separator();
void section_heading(const char* text);
// Returns true while the section is open. default_open applies on first use.
bool collapsing_section(const char* text, bool default_open = true);

} // namespace viewer::imgui_widgets
