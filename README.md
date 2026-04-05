# Padel Analytics — Power BI Dashboard

## Description
Dashboard BI pour l'analyse de l'écosystème padel professionnel.
Projet réalisé dans le cadre d'un cours de Business Intelligence.

## Pages
- Tournaments Page 1 : KPIs, Prize Money, Attendance, Scatter, Waterfall
- Tournaments Page 2 : Digital & Audience — Instagram, YouTube, TikTok

## Data Model
- 14 tables (10 dims + 4 facts)
- PostgreSQL local (padel_staging)
- Modèle en étoile — relations définies via clés étrangères

## DAX Measures
- Prize Money V3, Viewers V3, Occupation V3, Revenus Tickets V3
- ROI Tournoi (VAR chaining)
- Rank Tournoi Prize (RANKX)
- Engagement Moyen, Instagram Reach, YouTube Views, TikTok Videos

## SDG Alignment
- ODD 8 : Prize Money Total mesure la redistribution économique vers les athlètes
- ODD 17 : Engagement digital mesure les partenariats mondiaux

## Stack
- Power BI Desktop 2025
- PostgreSQL 16
- Python (Google Colab) pour ETL
- pgAdmin 4
