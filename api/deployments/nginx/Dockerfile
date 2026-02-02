# Utilise l'image officielle Nginx stable
FROM nginx:1.25.2-alpine

# Copie de la configuration nginx personnalisée
COPY nginx.conf /etc/nginx/nginx.conf

# Copie des certificats SSL
COPY certs /etc/nginx/certs

# Exposition des ports
EXPOSE 80 443

# Commande par défaut
CMD ["nginx", "-g", "daemon off;"]
