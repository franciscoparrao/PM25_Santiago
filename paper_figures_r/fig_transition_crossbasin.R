# Figura: descomposicion por regimen, cross-basin (Santiago + Salt Lake City)
# Dumbbell: la longitud del segmento Persistence -> XGBoost = valor que agrega el modelo.
# Mensaje: el segmento es largo en dias de transicion y corto en overall, en AMBAS cuencas.
setwd("/home/franciscoparrao/proyectos/Contaminacion/PM25_Santiago/paper_figures_r")
source("theme_paper.R")
setup_paper_theme("elsevier")
suppressPackageStartupMessages({library(ggplot2); library(dplyr); library(tidyr); library(patchwork)})

# SLC exacto desde el CSV; Santiago desde el manuscrito
slc <- read.csv("../data/processed/apr_2_slc_transition.csv")
get <- function(reg, col) slc[[col]][slc$regime==reg]
d <- tribble(
  ~basin, ~regime, ~Persistence, ~XGBoost,
  "Santiago",        "Overall (1-day)",     0.74, 0.76,
  "Santiago",        "Transition days",     0.03, 0.30,
  "Salt Lake City",  "Overall (1-day)",     get("overall","persistence_r2"),      get("overall","xgboost_r2"),
  "Salt Lake City",  "Transition days",     get("transition","persistence_r2"),   get("transition","xgboost_r2")
)
d$basin  <- factor(d$basin, levels=c("Santiago","Salt Lake City"))
d$regime <- factor(d$regime, levels=c("Transition days","Overall (1-day)"))  # transicion arriba
d$gap <- d$XGBoost - d$Persistence

col_p <- "#8C8C8C"; col_x <- "#0072B2"   # persistence gris, XGBoost Wong-blue

p <- ggplot(d, aes(y=regime)) +
  geom_vline(xintercept=0, linetype="dashed", color="grey60", linewidth=0.3) +
  geom_segment(aes(x=Persistence, xend=XGBoost, yend=regime),
               color="grey55", linewidth=0.9) +
  geom_point(aes(x=Persistence, color="Persistence"), size=2.6) +
  geom_point(aes(x=XGBoost, color="XGBoost"), size=2.6) +
  geom_text(aes(x=(Persistence+XGBoost)/2,
                label=sprintf("%+.2f", gap)),
            vjust=-0.9, size=2.5, color="grey25", family="Liberation Sans") +
  facet_wrap(~basin, ncol=1) +
  scale_color_manual(values=c(Persistence=col_p, XGBoost=col_x), name=NULL) +
  scale_x_continuous(limits=c(-0.35, 0.95), breaks=seq(-0.25,0.75,0.25),
                     expand=expansion(mult=c(0.02,0.06))) +
  labs(x=expression(R^2~"(1-day forecast)"), y=NULL) +
  theme_paper() +
  theme(legend.position="top",
        panel.grid.major.y=element_line(color="grey92", linewidth=0.3),
        strip.text=element_text(face="bold"))

save_paper(p, "figs/fig_transition_crossbasin.pdf", width_cm=12.0, height_cm=8.5)
cat("\n--- data ---\n"); print(as.data.frame(d))
