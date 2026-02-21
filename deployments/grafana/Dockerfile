FROM grafana/grafana:latest

ENV GF_DASHBOARDS_DEFAULT_HOME_DASHBOARD_PATH=/etc/grafana/provisioning/dashboards/api_monitoring.json

COPY provisioning/datasources/prometheus.yml /etc/grafana/provisioning/datasources/prometheus.yml

COPY provisioning/dashboards/dashboard.yml /etc/grafana/provisioning/dashboards/dashboard.yml

COPY provisioning/dashboards/*.json /etc/grafana/provisioning/dashboards

EXPOSE 3000