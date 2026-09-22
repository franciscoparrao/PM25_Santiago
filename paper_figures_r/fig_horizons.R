# Figura: skill por horizonte (XGBoost plano vs persistence colapsa)
setwd("/home/franciscoparrao/proyectos/Contaminacion/PM25_Santiago/paper_figures_r")
source("theme_paper.R"); setup_paper_theme("elsevier")
suppressPackageStartupMessages({library(ggplot2); library(dplyr); library(tidyr); library(ggrepel)})

d <- tribble(
  ~horizon, ~XGBoost, ~Persistence,
  1, 0.7638, 0.7411,
  3, 0.7406, 0.4438,
  7, 0.7437, 0.3659
)
dl <- pivot_longer(d, c(XGBoost, Persistence), names_to="model", values_to="r2")
dl$model <- factor(dl$model, levels=c("XGBoost","Persistence"))
gaps <- d %>% mutate(gap=round(100*(XGBoost-Persistence)/Persistence))

p <- ggplot(dl, aes(horizon, r2, color=model, group=model)) +
  geom_line(linewidth=0.8) +
  geom_point(size=2.6) +
  # anotar el gap creciente
  geom_segment(data=d, aes(x=horizon, xend=horizon, y=Persistence, yend=XGBoost),
               inherit.aes=FALSE, color="grey70", linewidth=0.3, linetype="dotted") +
  geom_label(data=gaps, aes(x=horizon, y=(XGBoost+Persistence)/2, label=paste0("+",gap,"%")),
             inherit.aes=FALSE, size=2.4, color="grey20", family="Liberation Sans",
             label.size=0, label.padding=unit(1,"pt"), fill="white") +
  scale_color_manual(values=c(XGBoost="#0072B2", Persistence="#8C8C8C"), name=NULL) +
  scale_x_continuous(breaks=c(1,3,7), expand=expansion(mult=c(0.05,0.05))) +
  scale_y_continuous(limits=c(0.3,0.85), breaks=seq(0.3,0.8,0.1)) +
  labs(x="Forecast horizon (days)", y=expression(R^2)) +
  theme_paper() + theme(legend.position="top",
        panel.grid.major.y=element_line(color="grey93", linewidth=0.3))
save_paper(p, "figs/fig_horizons.pdf", width_cm=9.0, height_cm=7.0)
cat("OK horizons\n")
