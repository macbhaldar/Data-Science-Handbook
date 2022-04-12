import numpy as np
import july
from july.utils import date_range

dates = date_range("2020-01-01", "2020-12-31")
data = np.random.randint(0, 14, len(dates))

# GitHub Activity like plot:
july.heatmap(dates, data, title='Github Activity', cmap="github")

# Daily heatmap for continuous data (with colourbar):
july.heatmap(
    dates,
    data, 
    cmap="golden", 
    colorbar=True, 
    title="Average temperatures: Oslo , Norway"
)

# Outline each month with month_grid=True
july.heatmap(dates=dates, 
             data=data, 
             cmap="Pastel1",
             month_grid=True, 
             horizontal=True,
             value_label=False,
             date_label=False,
             weekday_label=True,
             month_label=True, 
             year_label=True,
             colorbar=False,
             fontfamily="monospace",
             fontsize=12,
             title=None,
             titlesize="large",
             dpi=100)

# month or calendar plots:
july.calendar_plot(dates, data)

july.month_plot(dates, data, month=5) # This will plot only May.
