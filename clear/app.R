# app.R
library(shiny)
library(ggplot2)
library(tidyr)
library(dplyr)
library(brms)
library(bayesplot)
library(tidybayes)
library(ggridges)
library(ggpubr)

# UI remains the same
ui <- fluidPage(
  titlePanel("Bayesian Analysis of CLEAR Trial"),
  
  sidebarLayout(
    sidebarPanel(
      selectInput("prior_type", "Select Prior Type:",
                  choices = c("Skeptical", "Enthusiastic", "Pessimistic", "Meta-Analysis Based"),
                  selected = "Skeptical"),
      
      conditionalPanel(
        condition = "input.prior_type != 'Meta-Analysis Based'",
        sliderInput("mcid", "Minimal Clinically Important Difference (%):",
                    min = 0, max = 10, value = 2.8, step = 0.1)
      ),
      
      h4("Trial Summary:"),
      verbatimTextOutput("trial_summary"),
      
      downloadButton("download_plot", "Download Plot"),
      downloadButton("download_results", "Download Results")
    ),
    
    mainPanel(
      tabsetPanel(
        tabPanel("Distribution Plot", 
                 plotOutput("distribution_plot", height = "600px")),
        tabPanel("Forest Plot", 
                 plotOutput("forest_plot", height = "600px")),
        tabPanel("Results Summary",
                 verbatimTextOutput("analysis_summary"))
      )
    )
  )
)

# Server logic with fixed plotting
server <- function(input, output, session) {
  # Calculate the core parameters
  core_params <- reactive({
    # Trial data
    n_treatment <- 6992
    events_treatment <- 819
    n_control <- 6978
    events_control <- 927
    
    # Calculate observed log odds ratio and SE
    or <- (events_treatment / (n_treatment - events_treatment)) / 
      (events_control / (n_control - events_control))
    log_or <- log(or)
    se_log_or <- sqrt(1/events_treatment + 1/(n_treatment - events_treatment) + 
                        1/events_control + 1/(n_control - events_control))
    
    # Calculate MCID if needed
    baseline_risk <- events_control / n_control
    if (input$prior_type != "Meta-Analysis Based") {
      risk_treatment <- baseline_risk - input$mcid/100
      odds_control <- baseline_risk / (1 - baseline_risk)
      odds_treatment <- risk_treatment / (1 - risk_treatment)
      log_mcid <- log(odds_treatment / odds_control)
    } else {
      log_mcid <- log(0.8) # Default MCID for meta-analysis based prior
    }
    
    list(
      log_or = log_or,
      se_log_or = se_log_or,
      log_mcid = log_mcid,
      baseline_risk = baseline_risk
    )
  })
  
  # Calculate prior parameters
  prior_params <- reactive({
    params <- core_params()
    
    switch(input$prior_type,
           "Skeptical" = {
             z_skeptical <- qnorm(0.10)
             sigma_skeptical <- abs(params$log_mcid / z_skeptical)
             list(mu = 0, sigma = sigma_skeptical)
           },
           "Enthusiastic" = {
             z_enthusiastic <- qnorm(0.70)
             sigma_enthusiastic <- abs(params$log_mcid / z_enthusiastic)
             list(mu = params$log_mcid, sigma = sigma_enthusiastic)
           },
           "Pessimistic" = {
             z_pessimistic <- qnorm(0.30)
             sigma_pessimistic <- abs(params$log_mcid / z_pessimistic)
             list(mu = -params$log_mcid, sigma = sigma_pessimistic)
           },
           "Meta-Analysis Based" = {
             list(mu = -0.13, sigma = 0.05)
           })
  })
  
  # Calculate posterior parameters
  posterior_params <- reactive({
    params <- core_params()
    prior <- prior_params()
    
    var_prior <- prior$sigma^2
    var_data <- params$se_log_or^2
    var_post <- 1 / (1/var_prior + 1/var_data)
    mu_post <- var_post * (params$log_or/var_data + prior$mu/var_prior)
    
    list(mu = mu_post, sigma = sqrt(var_post))
  })
  
  # Distribution plot
  output$distribution_plot <- renderPlot({
    params <- core_params()
    prior <- prior_params()
    posterior <- posterior_params()
    
    # Generate x values for plotting
    x_vals <- seq(-1, 1, length.out = 500)
    
    # Create data frame for plotting
    plot_data <- data.frame(
      x = rep(x_vals, 3),
      y = c(
        dnorm(x_vals, prior$mu, prior$sigma),
        dnorm(x_vals, params$log_or, params$se_log_or),
        dnorm(x_vals, posterior$mu, posterior$sigma)
      ),
      Distribution = factor(rep(c("Prior", "Likelihood", "Posterior"), 
                                each = length(x_vals)),
                            levels = c("Prior", "Likelihood", "Posterior"))
    )
    
    ggplot(plot_data, aes(x = x, y = y, fill = Distribution)) +
      geom_area(alpha = 0.5, position = "identity") +
      geom_vline(xintercept = 0, linetype = "dashed") +
      scale_fill_manual(values = c("Prior" = "salmon", 
                                   "Likelihood" = "lightblue",
                                   "Posterior" = "darkgreen")) +
      labs(x = "Log Odds Ratio",
           y = "Density",
           title = paste("Distribution Plot -", input$prior_type, "Prior"),
           subtitle = "Prior, Likelihood, and Posterior Distributions") +
      theme_minimal() +
      theme(
        text = element_text(size = 14),
        plot.title = element_text(face = "bold"),
        legend.position = "bottom"
      )
  })
  
  # Trial summary output
  output$trial_summary <- renderText({
    paste("CLEAR Trial Data:\n",
          "Treatment group: 819/6992 events\n",
          "Control group: 927/6978 events")
  })
  
  # Add this code after the distribution_plot output in the server function:
  
  # Forest plot
  output$forest_plot <- renderPlot({
    # Define study data
    study_data <- data.frame(
      study = c("CLEAR Wisdom", "CLEAR Serenity", "CLEAR Harmony", "CLEAR Outcomes"),
      n_treatment = c(522, 234, 1487, 6992),
      events_treatment = c(32, 9, 68, 819),
      n_control = c(257, 111, 742, 6978),
      events_control = c(21, 1, 42, 927)
    )
    
    # Calculate log odds ratios and standard errors
    study_data <- study_data %>%
      mutate(
        or = (events_treatment/n_treatment)/(events_control/n_control),
        log_or = log(or),
        se = sqrt(1/events_treatment + 1/(n_treatment - events_treatment) + 
                    1/events_control + 1/(n_control - events_control)),
        lower_ci = log_or - 1.96 * se,
        upper_ci = log_or + 1.96 * se,
        # Convert to odds ratios for display
        or_display = exp(log_or),
        lower_ci_display = exp(lower_ci),
        upper_ci_display = exp(upper_ci)
      )
    
    # Add row for meta-analysis result (from your previous analysis)
    meta_row <- data.frame(
      study = "Meta-Analysis",
      log_or = -0.13,  # From your meta-analysis
      se = 0.05,       # From your meta-analysis
      lower_ci = -0.13 - 1.96 * 0.05,
      upper_ci = -0.13 + 1.96 * 0.05,
      or_display = exp(-0.13),
      lower_ci_display = exp(-0.13 - 1.96 * 0.05),
      upper_ci_display = exp(-0.13 + 1.96 * 0.05)
    )
    
    # Combine study data with meta-analysis
    plot_data <- bind_rows(study_data, meta_row)
    
    # Create the forest plot
    ggplot(plot_data, aes(y = reorder(study, log_or))) +
      # Add vertical line for null effect
      geom_vline(xintercept = 1, linetype = "dashed", color = "gray50") +
      # Add confidence intervals
      geom_errorbarh(aes(xmin = lower_ci_display, xmax = upper_ci_display), 
                     height = 0.2) +
      # Add point estimates
      geom_point(aes(x = or_display, size = ifelse(study == "Meta-Analysis", 2, 1),
                     shape = ifelse(study == "Meta-Analysis", "diamond", "circle"))) +
      # Customize appearance
      scale_x_continuous(trans = "log", breaks = c(0.5, 0.75, 1, 1.25, 1.5)) +
      scale_size_continuous(range = c(3, 5), guide = "none") +
      scale_shape_manual(values = c(diamond = 18, circle = 19), guide = "none") +
      # Labels
      labs(x = "Odds Ratio (95% CI)",
           y = NULL,
           title = "Forest Plot of CLEAR Trials",
           subtitle = "Including Meta-Analysis Result") +
      # Theme customization
      theme_minimal() +
      theme(
        text = element_text(size = 14),
        plot.title = element_text(face = "bold"),
        panel.grid.minor = element_blank(),
        panel.grid.major.y = element_blank(),
        axis.text = element_text(size = 12),
        plot.margin = margin(1, 1, 1, 1, "cm")
      ) +
      # Add annotations for odds ratios and CIs
      geom_text(aes(x = max(plot_data$upper_ci_display) * 1.2,
                    label = sprintf("%.2f (%.2f, %.2f)", 
                                    or_display, lower_ci_display, upper_ci_display)),
                size = 4)
  })
  
  # Add this to the results summary to include the meta-analysis results
  output$analysis_summary <- renderText({
    prior <- prior_params()
    posterior <- posterior_params()
    params <- core_params()
    
    paste0(
      "Analysis Results:\n\n",
      "Prior parameters:\n",
      sprintf("  Mean: %.3f\n", prior$mu),
      sprintf("  SD: %.3f\n\n", prior$sigma),
      "Posterior parameters:\n",
      sprintf("  Mean: %.3f\n", posterior$mu),
      sprintf("  SD: %.3f\n\n", posterior$sigma),
      "Probabilities:\n",
      sprintf("  P(OR < 1): %.1f%%\n", 
              100 * pnorm(0, posterior$mu, posterior$sigma)),
      if(input$prior_type != "Meta-Analysis Based") 
        sprintf("  P(OR < MCID): %.1f%%\n",
                100 * pnorm(params$log_mcid, posterior$mu, posterior$sigma))
      else "",
      "\nMeta-Analysis Results:\n",
      "OR: 0.88 (95% CI: 0.80, 0.96)"
    )
  })
  
  # Download handlers
  output$download_plot <- downloadHandler(
    filename = function() {
      paste("clear-trial-plot-", Sys.Date(), ".png", sep = "")
    },
    content = function(file) {
      ggsave(file, plot = last_plot(), width = 10, height = 8)
    }
  )
  
  output$download_results <- downloadHandler(
    filename = function() {
      paste("clear-trial-results-", Sys.Date(), ".txt", sep = "")
    },
    content = function(file) {
      writeLines(output$analysis_summary(), file)
    }
  )
}

# Run the app
shinyApp(ui = ui, server = server)