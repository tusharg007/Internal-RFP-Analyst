"""
Mock Internal Document Generator
Generates realistic fintech consulting documents as PDFs.
"""

import os
from fpdf import FPDF
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data" / "documents"

DOCUMENTS = [
    {
        "title": "Banking Sector Digital Audit 2024",
        "type": "Project Outline",
        "client_industry": "Banking & Financial Services",
        "sections": {
            "Executive Summary": (
                "This project delivered a comprehensive digital audit for a leading multinational bank. "
                "The engagement focused on evaluating the client's digital transformation maturity, "
                "identifying gaps in data governance, and recommending a modernization roadmap. "
                "Our team of 12 consultants worked across three workstreams over a 16-week engagement."
            ),
            "Objectives": (
                "1. Assess the current state of the client's core banking platform and digital channels.\n"
                "2. Evaluate data quality, lineage, and governance across 14 business units.\n"
                "3. Benchmark digital maturity against industry peers using proprietary frameworks.\n"
                "4. Deliver a prioritized transformation roadmap with ROI projections."
            ),
            "Technology Stack": (
                "- Cloud Platform: Microsoft Azure (Azure Data Factory, Azure SQL Database)\n"
                "- Analytics & Reporting: Power BI with DirectQuery for real-time dashboards\n"
                "- Data Processing: Python (Pandas, PySpark) for ETL pipelines\n"
                "- Audit Tools: ACL Analytics, IDEA Data Analysis\n"
                "- Collaboration: Microsoft Teams, SharePoint Online\n"
                "- Version Control: Azure DevOps with Git repositories"
            ),
            "Team Composition": (
                "- Engagement Partner: 1\n- Senior Manager: 1\n- Managers: 2\n"
                "- Senior Consultants: 4\n- Consultants: 3\n- Data Engineer: 1\n"
                "Total Team Size: 12 professionals"
            ),
            "Timeline & Milestones": (
                "- Phase 1 (Weeks 1-4): Discovery & current state assessment\n"
                "- Phase 2 (Weeks 5-8): Data quality analysis & gap identification\n"
                "- Phase 3 (Weeks 9-12): Benchmarking & peer analysis\n"
                "- Phase 4 (Weeks 13-16): Roadmap development & executive presentation\n"
                "Total Duration: 16 weeks"
            ),
            "Budget Range": "Estimated project cost: $850,000 - $1,100,000 USD",
            "Key Outcomes": (
                "- Identified 47 critical data governance gaps across the organization.\n"
                "- Achieved 23% improvement in data quality scores within the first quarter.\n"
                "- Delivered a 3-year digital transformation roadmap adopted by the board.\n"
                "- Reduced audit cycle time by 35% through automation of manual checks."
            ),
        },
    },
    {
        "title": "Healthcare Data Migration to Azure Cloud",
        "type": "Proposal",
        "client_industry": "Healthcare & Life Sciences",
        "sections": {
            "Executive Summary": (
                "Proposal for migrating a regional hospital network's on-premises data warehouse "
                "to Microsoft Azure. The project will ensure full HIPAA compliance, zero data loss, "
                "and improved analytics capabilities using Azure Databricks and Azure Synapse Analytics."
            ),
            "Objectives": (
                "1. Migrate 15TB of patient records and clinical data to Azure SQL Database.\n"
                "2. Implement HIPAA-compliant security controls including encryption at rest and in transit.\n"
                "3. Build a modern data lakehouse architecture using Azure Databricks.\n"
                "4. Enable self-service analytics for 200+ clinical staff using Power BI."
            ),
            "Technology Stack": (
                "- Cloud Platform: Microsoft Azure (Azure SQL, Azure Databricks, Azure Data Lake Storage Gen2)\n"
                "- ETL/ELT: Azure Data Factory with Mapping Data Flows\n"
                "- Analytics: Azure Synapse Analytics, Power BI Premium\n"
                "- Security: Azure Key Vault, Azure Active Directory, Microsoft Defender for Cloud\n"
                "- Compliance: HIPAA BAA, SOC 2 Type II controls\n"
                "- Monitoring: Azure Monitor, Log Analytics Workspace"
            ),
            "Team Composition": (
                "- Project Lead: 1\n- Cloud Architects: 2\n- Data Engineers: 3\n"
                "- Security Specialist: 1\n- HIPAA Compliance Officer: 1\n- QA Engineers: 2\n"
                "Total Team Size: 10 professionals"
            ),
            "Timeline & Milestones": (
                "- Phase 1 (Weeks 1-3): Architecture design & compliance planning\n"
                "- Phase 2 (Weeks 4-10): Data migration & validation\n"
                "- Phase 3 (Weeks 11-14): Analytics layer & dashboard development\n"
                "- Phase 4 (Weeks 15-18): UAT, security audit & go-live\n"
                "Total Duration: 18 weeks"
            ),
            "Budget Range": "Estimated project cost: $1,200,000 - $1,500,000 USD",
            "Key Outcomes": (
                "- Projected 40% reduction in infrastructure costs over 3 years.\n"
                "- Real-time clinical dashboards reducing report generation from 2 days to 15 minutes.\n"
                "- Full HIPAA compliance certification upon project completion.\n"
                "- Scalable architecture supporting 5x data growth without re-platforming."
            ),
        },
    },
    {
        "title": "Retail Supply Chain Analytics Platform",
        "type": "Case Study",
        "client_industry": "Retail & Consumer Goods",
        "sections": {
            "Executive Summary": (
                "Designed and deployed an end-to-end supply chain analytics platform for a Fortune 500 "
                "retailer. The platform provides real-time visibility into inventory levels, demand "
                "forecasting, and supplier performance across 2,000+ store locations."
            ),
            "Objectives": (
                "1. Build a centralized data warehouse consolidating data from 12 ERP systems.\n"
                "2. Implement ML-powered demand forecasting with 95%+ accuracy.\n"
                "3. Create real-time dashboards for supply chain KPIs.\n"
                "4. Reduce stockout rates by 30% and overstock by 25%."
            ),
            "Technology Stack": (
                "- Data Warehouse: Snowflake (Enterprise Edition)\n"
                "- Visualization: Tableau Server with embedded analytics\n"
                "- ML Pipeline: Python (scikit-learn, Prophet) on Databricks\n"
                "- Orchestration: Apache Airflow for pipeline scheduling\n"
                "- Data Integration: Fivetran for automated source ingestion\n"
                "- Infrastructure: AWS (S3, EC2, Lambda)"
            ),
            "Team Composition": (
                "- Engagement Lead: 1\n- Data Architects: 2\n- ML Engineers: 2\n"
                "- Tableau Developers: 2\n- Data Engineers: 3\n- Business Analysts: 2\n"
                "Total Team Size: 12 professionals"
            ),
            "Timeline & Milestones": (
                "- Phase 1 (Weeks 1-6): Data source inventory & warehouse design\n"
                "- Phase 2 (Weeks 7-14): ETL development & data integration\n"
                "- Phase 3 (Weeks 15-20): ML model development & validation\n"
                "- Phase 4 (Weeks 21-24): Dashboard development & UAT\n"
                "Total Duration: 24 weeks"
            ),
            "Budget Range": "Estimated project cost: $1,800,000 - $2,200,000 USD",
            "Key Outcomes": (
                "- Reduced stockout rates by 34%, exceeding the 30% target.\n"
                "- Demand forecasting accuracy reached 96.2% at SKU level.\n"
                "- Saved $12M annually in inventory carrying costs.\n"
                "- Dashboard adoption rate of 89% among supply chain managers."
            ),
        },
    },
    {
        "title": "Insurance Claims Processing Automation",
        "type": "RFP Response",
        "client_industry": "Insurance",
        "sections": {
            "Executive Summary": (
                "Response to RFP for automating the end-to-end claims processing workflow for a "
                "mid-tier insurance provider. The solution leverages RPA, AI document extraction, "
                "and Azure cloud services to reduce claims processing time by 60%."
            ),
            "Objectives": (
                "1. Automate document intake and classification for 15 claim types.\n"
                "2. Implement intelligent data extraction from unstructured claim forms.\n"
                "3. Build automated fraud detection rules engine.\n"
                "4. Reduce average claims processing time from 14 days to 5 days."
            ),
            "Technology Stack": (
                "- RPA Platform: UiPath Enterprise with Orchestrator\n"
                "- AI/ML: Azure Cognitive Services (Form Recognizer, Custom Vision)\n"
                "- Serverless: Azure Functions for event-driven processing\n"
                "- Database: Azure Cosmos DB for claims data store\n"
                "- Integration: Azure Service Bus for message queuing\n"
                "- Monitoring: Application Insights, Power Automate alerts"
            ),
            "Team Composition": (
                "- Solution Architect: 1\n- RPA Developers: 3\n- AI/ML Engineers: 2\n"
                "- Azure Cloud Engineer: 1\n- Business Analyst: 1\n- Test Lead: 1\n"
                "Total Team Size: 9 professionals"
            ),
            "Timeline & Milestones": (
                "- Phase 1 (Weeks 1-4): Process mapping & bot design\n"
                "- Phase 2 (Weeks 5-10): RPA development & AI model training\n"
                "- Phase 3 (Weeks 11-14): Integration & end-to-end testing\n"
                "- Phase 4 (Weeks 15-16): Pilot deployment & hypercare\n"
                "Total Duration: 16 weeks"
            ),
            "Budget Range": "Estimated project cost: $650,000 - $800,000 USD",
            "Key Outcomes": (
                "- Projected 60% reduction in claims processing time.\n"
                "- 85% straight-through processing rate for standard claims.\n"
                "- Estimated annual savings of $3.2M in operational costs.\n"
                "- Fraud detection rate improvement from 12% to 34%."
            ),
        },
    },
    {
        "title": "Telecom Network Optimization with AI",
        "type": "Project Outline",
        "client_industry": "Telecommunications",
        "sections": {
            "Executive Summary": (
                "AI-driven network optimization project for a Tier-1 telecom operator. "
                "Deployed machine learning models for predictive maintenance, capacity planning, "
                "and anomaly detection across 50,000+ network nodes."
            ),
            "Objectives": (
                "1. Build predictive maintenance models reducing unplanned outages by 40%.\n"
                "2. Implement real-time anomaly detection on network traffic patterns.\n"
                "3. Optimize capacity planning using historical usage data and ML forecasting.\n"
                "4. Create an executive dashboard for network health monitoring."
            ),
            "Technology Stack": (
                "- ML Framework: TensorFlow 2.x, Keras for deep learning models\n"
                "- Cloud: Azure Machine Learning for model training & deployment\n"
                "- Edge Computing: Azure IoT Edge for on-premise inference\n"
                "- Stream Processing: Apache Kafka, Azure Stream Analytics\n"
                "- Visualization: Grafana for real-time network monitoring\n"
                "- Data Storage: Azure Data Lake Storage Gen2, Delta Lake"
            ),
            "Team Composition": (
                "- Technical Lead: 1\n- ML Scientists: 3\n- Data Engineers: 2\n"
                "- IoT Specialists: 2\n- DevOps Engineer: 1\n- Domain Expert: 1\n"
                "Total Team Size: 10 professionals"
            ),
            "Timeline & Milestones": (
                "- Phase 1 (Weeks 1-4): Data collection & feature engineering\n"
                "- Phase 2 (Weeks 5-12): Model development & training\n"
                "- Phase 3 (Weeks 13-18): Edge deployment & integration\n"
                "- Phase 4 (Weeks 19-22): Monitoring, tuning & handover\n"
                "Total Duration: 22 weeks"
            ),
            "Budget Range": "Estimated project cost: $1,400,000 - $1,700,000 USD",
            "Key Outcomes": (
                "- Reduced unplanned network outages by 43%.\n"
                "- Anomaly detection system catches 92% of issues before customer impact.\n"
                "- Saved $8.5M annually in maintenance and downtime costs.\n"
                "- Capacity planning accuracy improved from 71% to 94%."
            ),
        },
    },
    {
        "title": "Government Tax Filing Modernization",
        "type": "Proposal",
        "client_industry": "Government & Public Sector",
        "sections": {
            "Executive Summary": (
                "Proposal to modernize a national tax filing system from legacy mainframe to a "
                "cloud-native microservices architecture. The solution will improve citizen experience, "
                "reduce processing time, and enable real-time tax computation."
            ),
            "Objectives": (
                "1. Migrate legacy COBOL-based tax engine to cloud-native microservices.\n"
                "2. Build a modern citizen-facing web portal with mobile responsiveness.\n"
                "3. Implement real-time tax computation and refund processing.\n"
                "4. Ensure compliance with government security standards (FedRAMP)."
            ),
            "Technology Stack": (
                "- Backend: Java Spring Boot microservices, SAP HANA for tax computation\n"
                "- Frontend: React.js with accessibility compliance (WCAG 2.1 AA)\n"
                "- Cloud: Azure Government Cloud\n"
                "- API Gateway: Azure API Management\n"
                "- CI/CD: Azure DevOps Pipelines\n"
                "- Security: Azure Sentinel, Multi-factor Authentication"
            ),
            "Team Composition": (
                "- Program Manager: 1\n- Solution Architects: 2\n- Java Developers: 5\n"
                "- Frontend Developers: 3\n- Security Consultants: 2\n- QA Team: 3\n"
                "Total Team Size: 16 professionals"
            ),
            "Timeline & Milestones": (
                "- Phase 1 (Months 1-2): Legacy system analysis & architecture design\n"
                "- Phase 2 (Months 3-6): Microservices development & API layer\n"
                "- Phase 3 (Months 7-9): Frontend development & integration\n"
                "- Phase 4 (Months 10-12): Security audit, UAT & phased rollout\n"
                "Total Duration: 12 months"
            ),
            "Budget Range": "Estimated project cost: $4,500,000 - $5,800,000 USD",
            "Key Outcomes": (
                "- Projected 70% reduction in tax return processing time.\n"
                "- Expected to handle 10x peak load compared to legacy system.\n"
                "- Mobile-responsive portal serving 50M+ citizens.\n"
                "- Annual infrastructure cost reduction of 45%."
            ),
        },
    },
    {
        "title": "Pharma Clinical Trial Data Platform",
        "type": "Case Study",
        "client_industry": "Pharmaceutical & Life Sciences",
        "sections": {
            "Executive Summary": (
                "Built a unified clinical trial data management platform for a top-10 pharmaceutical "
                "company. The platform integrates data from 200+ active clinical trials across "
                "40 countries, ensuring FDA 21 CFR Part 11 compliance."
            ),
            "Objectives": (
                "1. Consolidate clinical trial data from 8 disparate CTMS systems.\n"
                "2. Build automated regulatory reporting for FDA and EMA submissions.\n"
                "3. Implement data quality monitoring with automated anomaly alerting.\n"
                "4. Reduce time-to-insight for clinical operations from weeks to hours."
            ),
            "Technology Stack": (
                "- Data Warehouse: Amazon Redshift (RA3 nodes)\n"
                "- Orchestration: Apache Airflow on Amazon MWAA\n"
                "- Data Quality: Great Expectations for automated validation\n"
                "- Compliance: FDA 21 CFR Part 11, GDPR, ICH E6(R2) GCP\n"
                "- Reporting: SAS Visual Analytics, custom Python dashboards\n"
                "- Infrastructure: AWS (S3, Glue, Lambda, Step Functions)"
            ),
            "Team Composition": (
                "- Engagement Partner: 1\n- Data Architects: 2\n- ETL Developers: 4\n"
                "- Regulatory Compliance SME: 1\n- Data Quality Analysts: 2\n- SAS Developers: 2\n"
                "Total Team Size: 12 professionals"
            ),
            "Timeline & Milestones": (
                "- Phase 1 (Months 1-2): Data source profiling & compliance mapping\n"
                "- Phase 2 (Months 3-5): Data warehouse design & ETL development\n"
                "- Phase 3 (Months 6-7): Data quality framework & validation\n"
                "- Phase 4 (Months 8-9): Reporting layer & regulatory submission automation\n"
                "Total Duration: 9 months"
            ),
            "Budget Range": "Estimated project cost: $2,800,000 - $3,400,000 USD",
            "Key Outcomes": (
                "- Reduced clinical data reconciliation time by 75%.\n"
                "- Automated 85% of routine regulatory submissions.\n"
                "- Achieved full FDA 21 CFR Part 11 compliance certification.\n"
                "- Time-to-insight reduced from 3 weeks to 4 hours."
            ),
        },
    },
    {
        "title": "Energy Sector ESG Reporting Dashboard",
        "type": "RFP Response",
        "client_industry": "Energy & Utilities",
        "sections": {
            "Executive Summary": (
                "RFP response for building an ESG (Environmental, Social, Governance) reporting "
                "platform for a multinational energy company. The solution automates ESG data "
                "collection, calculates carbon footprint metrics, and generates investor-ready reports."
            ),
            "Objectives": (
                "1. Automate ESG data collection from 50+ operational sites globally.\n"
                "2. Build carbon footprint calculation engine aligned with GHG Protocol.\n"
                "3. Create investor-ready ESG dashboards compliant with GRI and SASB standards.\n"
                "4. Enable scenario modeling for net-zero pathway planning."
            ),
            "Technology Stack": (
                "- Analytics Engine: Azure Synapse Analytics for large-scale data processing\n"
                "- Visualization: Power BI Embedded with custom visuals\n"
                "- Data Integration: Azure Data Factory with 50+ source connectors\n"
                "- Calculation Engine: Python-based GHG calculation models\n"
                "- Standards: GRI Standards, SASB, TCFD, EU Taxonomy\n"
                "- Cloud: Microsoft Azure (Central deployment)"
            ),
            "Team Composition": (
                "- ESG Domain Lead: 1\n- Data Engineers: 3\n- Power BI Developers: 2\n"
                "- Python Developers: 2\n- Sustainability Consultant: 1\n- Project Manager: 1\n"
                "Total Team Size: 10 professionals"
            ),
            "Timeline & Milestones": (
                "- Phase 1 (Weeks 1-4): ESG framework selection & data mapping\n"
                "- Phase 2 (Weeks 5-10): Data pipeline & calculation engine\n"
                "- Phase 3 (Weeks 11-16): Dashboard development & report templates\n"
                "- Phase 4 (Weeks 17-20): Validation, audit trail & deployment\n"
                "Total Duration: 20 weeks"
            ),
            "Budget Range": "Estimated project cost: $950,000 - $1,200,000 USD",
            "Key Outcomes": (
                "- Projected 80% reduction in ESG reporting preparation time.\n"
                "- Automated carbon footprint calculation across Scope 1, 2, and 3 emissions.\n"
                "- Board-ready ESG dashboards updated in real-time.\n"
                "- Compliance with 4 major ESG reporting frameworks."
            ),
        },
    },
    {
        "title": "Financial Services Anti-Fraud Detection System",
        "type": "Project Outline",
        "client_industry": "Financial Services & Fintech",
        "sections": {
            "Executive Summary": (
                "Designed and deployed an AI-powered fraud detection system for a digital payments "
                "fintech processing 5M+ transactions daily. The system combines traditional ML models "
                "with LLM-powered investigation agents for comprehensive fraud prevention."
            ),
            "Objectives": (
                "1. Build real-time fraud scoring for card-not-present transactions.\n"
                "2. Implement an LLM-powered fraud investigation agent for complex cases.\n"
                "3. Reduce false positive rates by 50% while maintaining detection accuracy.\n"
                "4. Create a case management dashboard for fraud analysts."
            ),
            "Technology Stack": (
                "- ML Models: XGBoost, LightGBM for real-time fraud scoring\n"
                "- AI Agent: LangGraph with ReAct pattern for fraud investigation\n"
                "- Streaming: Apache Kafka, Flink for real-time event processing\n"
                "- Database: PostgreSQL, Redis for feature store\n"
                "- Cloud: Azure Kubernetes Service for model serving\n"
                "- Monitoring: MLflow for model versioning, Prometheus + Grafana"
            ),
            "Team Composition": (
                "- ML Lead: 1\n- ML Engineers: 3\n- Backend Developers: 2\n"
                "- Data Engineers: 2\n- DevOps Engineer: 1\n- Fraud Domain Expert: 1\n"
                "Total Team Size: 10 professionals"
            ),
            "Timeline & Milestones": (
                "- Phase 1 (Weeks 1-4): Feature engineering & historical data analysis\n"
                "- Phase 2 (Weeks 5-10): ML model development & LLM agent creation\n"
                "- Phase 3 (Weeks 11-16): Real-time pipeline & integration\n"
                "- Phase 4 (Weeks 17-20): A/B testing, monitoring & production rollout\n"
                "Total Duration: 20 weeks"
            ),
            "Budget Range": "Estimated project cost: $1,600,000 - $2,000,000 USD",
            "Key Outcomes": (
                "- Fraud detection rate increased from 67% to 94%.\n"
                "- False positive rate reduced by 52%, saving analyst investigation time.\n"
                "- LLM agent automates 70% of routine fraud investigations.\n"
                "- System processes 5M+ transactions daily with <100ms latency."
            ),
        },
    },
    {
        "title": "Manufacturing IoT Predictive Maintenance Platform",
        "type": "Proposal",
        "client_industry": "Manufacturing & Industrial",
        "sections": {
            "Executive Summary": (
                "Proposal for implementing an IoT-based predictive maintenance platform for a global "
                "automotive manufacturer. The solution connects 10,000+ sensors across 8 factories "
                "to predict equipment failures 72 hours in advance."
            ),
            "Objectives": (
                "1. Deploy IoT sensor network across 8 manufacturing facilities.\n"
                "2. Build predictive maintenance ML models with 72-hour failure prediction.\n"
                "3. Create a unified operations dashboard for plant managers.\n"
                "4. Reduce unplanned downtime by 45% and maintenance costs by 30%."
            ),
            "Technology Stack": (
                "- IoT Platform: Azure IoT Hub with IoT Edge for local processing\n"
                "- Time Series: Azure Time Series Insights for sensor data analysis\n"
                "- ML: Azure Machine Learning, Python (scikit-learn, TensorFlow)\n"
                "- Data Storage: Azure Data Lake Storage, Azure SQL Database\n"
                "- Visualization: Power BI with real-time streaming datasets\n"
                "- Edge: NVIDIA Jetson devices for on-premise ML inference"
            ),
            "Team Composition": (
                "- IoT Architect: 1\n- IoT Engineers: 3\n- ML Engineers: 2\n"
                "- Data Engineers: 2\n- Power BI Developer: 1\n- Industrial Engineer: 1\n"
                "Total Team Size: 10 professionals"
            ),
            "Timeline & Milestones": (
                "- Phase 1 (Months 1-2): Sensor audit & IoT architecture design\n"
                "- Phase 2 (Months 3-5): Sensor deployment & data pipeline\n"
                "- Phase 3 (Months 6-8): ML model development & validation\n"
                "- Phase 4 (Months 9-10): Dashboard, alerting & phased rollout\n"
                "Total Duration: 10 months"
            ),
            "Budget Range": "Estimated project cost: $2,200,000 - $2,800,000 USD",
            "Key Outcomes": (
                "- Projected 45% reduction in unplanned equipment downtime.\n"
                "- 72-hour advance warning for 89% of critical equipment failures.\n"
                "- Estimated annual savings of $15M across all facilities.\n"
                "- ROI breakeven within 14 months of deployment."
            ),
        },
    },
]


class PDFDocument(FPDF):
    """Custom PDF with professional formatting."""

    def __init__(self, doc_title, doc_type, industry):
        super().__init__()
        self.doc_title = doc_title
        self.doc_type = doc_type
        self.industry = industry

    def header(self):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, f"CONFIDENTIAL | {self.doc_type.upper()} | {self.industry}", align="R")
        self.ln(12)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}} | Internal Use Only", align="C")

    def add_title_page(self):
        self.add_page()
        self.ln(40)
        self.set_font("Helvetica", "B", 24)
        self.set_text_color(30, 30, 80)
        self.multi_cell(0, 12, self.doc_title, align="C")
        self.ln(10)
        self.set_font("Helvetica", "", 14)
        self.set_text_color(100, 100, 100)
        self.cell(0, 10, f"Document Type: {self.doc_type}", align="C")
        self.ln(8)
        self.cell(0, 10, f"Industry: {self.industry}", align="C")
        self.ln(8)
        self.cell(0, 10, "Classification: Internal / Confidential", align="C")
        self.ln(30)
        self.set_draw_color(80, 80, 180)
        self.set_line_width(0.5)
        self.line(30, self.get_y(), 180, self.get_y())

    def add_section(self, title, content):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(30, 30, 80)
        self.cell(0, 10, title)
        self.ln(8)
        self.set_draw_color(80, 80, 180)
        self.set_line_width(0.3)
        self.line(10, self.get_y(), 60, self.get_y())
        self.ln(4)
        self.set_font("Helvetica", "", 11)
        self.set_text_color(50, 50, 50)
        self.multi_cell(0, 6, content)
        self.ln(6)


def generate_all_documents():
    """Generate all mock PDF documents."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for i, doc in enumerate(DOCUMENTS, 1):
        pdf = PDFDocument(doc["title"], doc["type"], doc["client_industry"])
        pdf.alias_nb_pages()
        pdf.add_title_page()
        pdf.add_page()

        for section_title, section_content in doc["sections"].items():
            if pdf.get_y() > 240:
                pdf.add_page()
            pdf.add_section(section_title, section_content)

        filename = f"{i:02d}_{doc['title'].replace(' ', '_').replace('/', '_')}.pdf"
        filepath = DATA_DIR / filename
        pdf.output(str(filepath))
        print(f"[{i}/10] Generated: {filename}")

    print(f"\nAll {len(DOCUMENTS)} documents saved to: {DATA_DIR}")


if __name__ == "__main__":
    generate_all_documents()
