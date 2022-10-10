library(ggplot2)
library(dplyr)

seq(-10, 10, 0.01) %>%
    expand.grid(x = ., y = .) %>%
    ggplot(aes(x = x + y + pi * sin(y), y = y + pi * sin(x))) +
    geom_point(
        alpha = 0.15, pch = 20, size = 0.2,
        aes(col = abs(sin(x)))
    ) +
    coord_polar() +
    theme_minimal() +
    theme(
        panel.grid = element_blank(),
        panel.background = element_blank(),
        axis.text = element_blank(),
        axis.ticks = element_blank(),
        axis.title = element_blank(),
        legend.position = "none"
    )