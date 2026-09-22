# Figura: comparacion like-for-like sobre 493 origenes identicos (reemplaza el 7-panel)
# Mensaje: sobre puntos identicos XGBoost gana en CADA metrica.
setwd("/home/franciscoparrao/proyectos/Contaminacion/PM25_Santiago/paper_figures_r")
source("theme_paper.R"); setup_paper_theme("elsevier")
suppressPackageStartupMessages({library(ggplot2); library(dplyr); library(tidyr); library(forcats)})

d <- read.csv("../data/processed/apr_1_3_like_for_like.csv")
dl <- d %>% select(model, r2, rmse, mae, mape) %>%
  pivot_longer(-model, names_to="metric", values_to="value")
# etiquetas de metrica y orden (R2: mas alto mejor; errores: mas bajo mejor)
lab <- c(r2="R^2~'(higher better)'", rmse="RMSE~'('*mu*g~m^{-3}*')'",
         mae="MAE~'('*mu*g~m^{-3}*')'", mape="MAPE~'(%)'")
dl$metric <- factor(dl$metric, levels=names(lab), labels=lab)
dl$model  <- factor(dl$model, levels=c("XGBoost","ARIMA","Prophet","Persistence"))
dl$hl <- ifelse(dl$model=="XGBoost","XGBoost","other")

p <- ggplot(dl, aes(fct_rev(model), value, fill=hl)) +
  geom_col(width=0.68, color="grey25", linewidth=0.2) +
  geom_text(aes(label=sprintf("%.2g", value)), hjust=-0.15, size=2.2, family="Liberation Sans") +
  coord_flip() +
  facet_wrap(~metric, scales="free_x", nrow=1, labeller=label_parsed) +
  scale_fill_manual(values=c(XGBoost="#0072B2", other="#B0B0B0"), guide="none") +
  scale_y_continuous(expand=expansion(mult=c(0,0.18))) +
  labs(x=NULL, y=NULL) +
  theme_paper() +
  theme(panel.grid.major.x=element_line(color="grey93", linewidth=0.3),
        strip.text=element_text(size=8))
save_paper(p, "figs/fig_likeforlike.pdf", width_cm=18.0, height_cm=5.2)
cat("OK like-for-like\n")
