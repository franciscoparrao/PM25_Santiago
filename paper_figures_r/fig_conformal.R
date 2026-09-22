# Figura: calibracion de intervalos por conformal (cobertura vs nominal 90%)
# Mensaje: el conformal NORMALIZADO recupera cobertura global Y en el tail de episodios.
setwd("/home/franciscoparrao/proyectos/Contaminacion/PM25_Santiago/paper_figures_r")
source("theme_paper.R")
setup_paper_theme("elsevier")
suppressPackageStartupMessages({library(ggplot2); library(dplyr); library(tidyr)})

# valores de la Tabla de conformal del manuscrito (tab:conformal)
d <- tribble(
  ~method,                        ~Overall, ~Episodes,
  "Quantile\nregression",           61.0,    NA,
  "Split\nconformal",               88.5,    50.4,
  "Level-normalized\nconformal",    87.0,    78.5
)
d$method <- factor(d$method, levels=d$method)
dl <- pivot_longer(d, c(Overall, Episodes), names_to="regime", values_to="coverage") %>%
      filter(!is.na(coverage))
dl$regime <- factor(dl$regime, levels=c("Overall","Episodes"),
                    labels=c("Overall", "Alert episodes (≥ 80)"))

p <- ggplot(dl, aes(method, coverage, fill=regime)) +
  geom_col(position=position_dodge(width=0.7), width=0.62, color="grey20", linewidth=0.2) +
  geom_hline(yintercept=90, linetype="dashed", color="#D55E00", linewidth=0.5) +
  annotate("text", x=0.55, y=92.5, label="nominal 90%", hjust=0, size=2.5,
           color="#D55E00", family="Liberation Sans") +
  geom_text(aes(label=sprintf("%.0f", coverage)),
            position=position_dodge(width=0.7), vjust=-0.5, size=2.5,
            family="Liberation Sans", color="grey20") +
  scale_fill_manual(values=c("Overall"="#56B4E9","Alert episodes (≥ 80)"="#D55E00"),
                    name=NULL) +
  scale_y_continuous(limits=c(0,100), breaks=seq(0,100,25),
                     expand=expansion(mult=c(0,0.04))) +
  labs(x=NULL, y="Empirical coverage of 90% interval (%)") +
  theme_paper() +
  theme(legend.position="top")

save_paper(p, "figs/fig_conformal.pdf", width_cm=11.0, height_cm=7.5)
cat("\nOK conformal\n")
