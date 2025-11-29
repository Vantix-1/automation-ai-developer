import React, { useState } from 'react';
import { Check, Circle, Clock, BookOpen, Code, Rocket, Brain, Database } from 'lucide-react';

export default function Phase2Roadmap() {
  const [expandedWeek, setExpandedWeek] = useState(null);

  const weeks = [
    {
      week: "Week 1-2",
      title: "OpenAI API Mastery",
      icon: Brain,
      color: "bg-blue-500",
      focus: "API Fundamentals & Prompt Engineering",
      dailyTasks: [
        {
          day: "Days 1-2",
          tasks: [
            "Set up development environment (Python 3.11+, VS Code)",
            "Install: pip install openai python-dotenv",
            "Create OpenAI account & get API key ($5 credit)",
            "Complete OpenAI Quickstart tutorial",
            "Build: hello_openai.py - First API call"
          ]
        },
        {
          day: "Days 3-5",
          tasks: [
            "Complete DeepLearning.AI: 'ChatGPT Prompt Engineering' (2 hrs)",
            "Learn: Chat completions, system/user/assistant roles",
            "Build: ai_chat_bot.py - Interactive CLI chatbot",
            "Add: Conversation history & context management",
            "Implement: Temperature, max_tokens parameters"
          ]
        },
        {
          day: "Days 6-8",
          tasks: [
            "Learn: Streaming responses for real-time output",
            "Build: ai_content_summarizer.py",
            "Practice: Different prompt patterns (few-shot, chain-of-thought)",
            "Add: Token counting & cost estimation",
            "Create: Prompt template library"
          ]
        },
        {
          day: "Days 9-10",
          tasks: [
            "Learn: Function calling / tool use",
            "Build: weather_assistant.py with function calls",
            "Experiment: JSON mode for structured outputs",
            "Study: Error handling & rate limiting",
            "Document: Your learnings in README"
          ]
        }
      ],
      resources: [
        "OpenAI Cookbook (cookbook.openai.com)",
        "DeepLearning.AI: ChatGPT Prompt Engineering (free)",
        "OpenAI API Documentation"
      ],
      deliverables: [
        "ai_chat_bot.py - Working chatbot",
        "ai_content_summarizer.py - Text summarization",
        "weather_assistant.py - Function calling demo",
        "prompts.md - Prompt engineering notes"
      ]
    },
    {
      week: "Week 3-4",
      title: "LangChain Fundamentals",
      icon: Code,
      color: "bg-purple-500",
      focus: "Chains, Agents & Memory",
      dailyTasks: [
        {
          day: "Days 11-13",
          tasks: [
            "Install: pip install langchain langchain-openai",
            "Complete: DeepLearning.AI 'LangChain for LLM Development' (3 hrs)",
            "Learn: LangChain architecture (models, prompts, chains)",
            "Build: simple_chain.py - Your first LangChain app",
            "Practice: PromptTemplates & output parsers"
          ]
        },
        {
          day: "Days 14-16",
          tasks: [
            "Learn: Sequential chains & routing",
            "Build: multi_step_assistant.py",
            "Implement: LLMChain, SimpleSequentialChain",
            "Add: Memory (ConversationBufferMemory)",
            "Practice: Context window management"
          ]
        },
        {
          day: "Days 17-19",
          tasks: [
            "Learn: LangChain Agents & Tools",
            "Build: research_agent.py with web search",
            "Implement: ReAct agent pattern",
            "Add: Custom tools for your use case",
            "Study: Agent decision-making process"
          ]
        },
        {
          day: "Days 20-21",
          tasks: [
            "Build: Conversational agent with persistent memory",
            "Implement: ConversationSummaryMemory",
            "Add: Chat history in JSON/SQLite",
            "Create: Multi-turn conversation demo",
            "Review & refactor your code"
          ]
        }
      ],
      resources: [
        "LangChain Documentation (python.langchain.com)",
        "DeepLearning.AI: LangChain Course (free)",
        "LangChain Cookbook on GitHub",
        "YouTube: Sam Witteveen LangChain tutorials"
      ],
      deliverables: [
        "simple_chain.py - Basic LangChain implementation",
        "multi_step_assistant.py - Sequential chains",
        "research_agent.py - Agent with tools",
        "conversational_bot.py - Chat with memory"
      ]
    },
    {
      week: "Week 5-6",
      title: "RAG Systems & Vector DBs",
      icon: Database,
      color: "bg-green-500",
      focus: "Document Q&A & Embeddings",
      dailyTasks: [
        {
          day: "Days 22-24",
          tasks: [
            "Learn: Embeddings & vector similarity",
            "Install: pip install chromadb sentence-transformers",
            "Build: embedding_demo.py - Understand embeddings",
            "Practice: Cosine similarity calculations",
            "Create: Simple semantic search engine"
          ]
        },
        {
          day: "Days 25-27",
          tasks: [
            "Learn: RAG architecture (Retrieval-Augmented Generation)",
            "Install: pip install pypdf unstructured",
            "Build: document_loader.py - Load & chunk documents",
            "Implement: Text splitting strategies",
            "Create: ChromaDB vector store"
          ]
        },
        {
          day: "Days 28-30",
          tasks: [
            "Build: rag_document_qa.py - Your first RAG system",
            "Implement: Document upload + query interface",
            "Add: Source attribution in responses",
            "Practice: Chunking strategies (size, overlap)",
            "Test: With multiple PDFs"
          ]
        },
        {
          day: "Days 31-33",
          tasks: [
            "Enhance: Add metadata filtering",
            "Implement: Hybrid search (keyword + semantic)",
            "Add: Re-ranking for better results",
            "Build: multi_doc_assistant.py",
            "Document: RAG best practices"
          ]
        }
      ],
      resources: [
        "Pinecone Learning Center (RAG tutorials)",
        "LangChain RAG Documentation",
        "ChromaDB Cookbook",
        "YouTube: AI Jason RAG videos"
      ],
      deliverables: [
        "rag_document_qa.py - Working RAG system",
        "multi_doc_assistant.py - Multi-document Q&A",
        "vector_search_demo.py - Semantic search",
        "rag_notes.md - RAG implementation guide"
      ]
    },
    {
      week: "Week 7-8",
      title: "Production AI APIs",
      icon: Rocket,
      color: "bg-orange-500",
      focus: "FastAPI & Deployment",
      dailyTasks: [
        {
          day: "Days 34-36",
          tasks: [
            "Learn: FastAPI basics (Official Tutorial - 4 hrs)",
            "Install: pip install fastapi uvicorn",
            "Build: hello_api.py - Your first API",
            "Practice: Path parameters, query params, request bodies",
            "Add: Pydantic models for validation"
          ]
        },
        {
          day: "Days 37-39",
          tasks: [
            "Build: ai_chat_api.py - Chatbot as REST API",
            "Implement: POST /chat endpoint",
            "Add: Streaming responses with Server-Sent Events",
            "Practice: Error handling & status codes",
            "Test: Using Postman or curl"
          ]
        },
        {
          day: "Days 40-42",
          tasks: [
            "Build: rag_api.py - RAG system as API",
            "Implement: /upload, /query endpoints",
            "Add: File upload handling",
            "Create: Simple HTML frontend",
            "Practice: CORS, security headers"
          ]
        },
        {
          day: "Days 43-45",
          tasks: [
            "Learn: Docker basics (Docker 101 tutorial)",
            "Create: Dockerfile for your AI API",
            "Build: Docker image & run container",
            "Add: Environment variables & secrets",
            "Deploy: Test locally with docker-compose"
          ]
        },
        {
          day: "Days 46-48",
          tasks: [
            "Build: flask_ai_dashboard.py (from roadmap)",
            "Implement: Streamlit alternative dashboard",
            "Add: Usage metrics & monitoring",
            "Create: API documentation with FastAPI auto-docs",
            "Final: Portfolio README with demos"
          ]
        }
      ],
      resources: [
        "FastAPI Official Tutorial (fastapi.tiangolo.com)",
        "Docker Getting Started (docs.docker.com)",
        "Streamlit Documentation (for dashboards)",
        "Real Python: Building APIs with FastAPI"
      ],
      deliverables: [
        "ai_chat_api.py - Production chatbot API",
        "rag_api.py - Document Q&A API",
        "flask_ai_dashboard.py - Monitoring dashboard",
        "Dockerfile - Containerized app",
        "README.md - Complete API documentation"
      ]
    }
  ];

  const setupChecklist = [
    "Python 3.11+ installed",
    "VS Code with Python extension",
    "Git for version control",
    "OpenAI API account ($5 credit)",
    "Virtual environment: python -m venv ai_env",
    "Install core libraries: pip install openai langchain chromadb fastapi",
    "Create .env file for API keys",
    "GitHub repo updated with Phase 2 folder"
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white p-8">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="mb-12 text-center">
          <div className="inline-block bg-gradient-to-r from-blue-500 to-purple-500 px-6 py-2 rounded-full text-sm font-semibold mb-4">
            PHASE 2: AI APIS & LLM INTEGRATION
          </div>
          <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
            8-Week Intensive Program
          </h1>
          <p className="text-xl text-gray-300 max-w-3xl mx-auto">
            Transform from Python developer to AI engineer. Build production-ready LLM applications, RAG systems, and intelligent APIs.
          </p>
          <div className="mt-6 text-sm text-gray-400">
            Started: January 2025 • Target Completion: March 2025
          </div>
        </div>

        {/* Setup Checklist */}
        <div className="bg-white/5 backdrop-blur-lg rounded-2xl p-8 mb-8 border border-white/10">
          <h2 className="text-2xl font-bold mb-6 flex items-center gap-3">
            <Check className="text-green-400" size={28} />
            Pre-Phase 2 Setup Checklist
          </h2>
          <div className="grid md:grid-cols-2 gap-4">
            {setupChecklist.map((item, idx) => (
              <div key={idx} className="flex items-start gap-3 bg-white/5 rounded-lg p-3">
                <Circle size={20} className="text-purple-400 mt-0.5 flex-shrink-0" />
                <span className="text-gray-300">{item}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Weekly Roadmap */}
        <div className="space-y-6">
          {weeks.map((week, idx) => {
            const Icon = week.icon;
            const isExpanded = expandedWeek === idx;
            
            return (
              <div key={idx} className="bg-white/5 backdrop-blur-lg rounded-2xl border border-white/10 overflow-hidden">
                <div 
                  className="p-6 cursor-pointer hover:bg-white/5 transition-all"
                  onClick={() => setExpandedWeek(isExpanded ? null : idx)}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4">
                      <div className={`${week.color} p-3 rounded-xl`}>
                        <Icon size={28} />
                      </div>
                      <div>
                        <div className="text-sm text-gray-400 mb-1">{week.week}</div>
                        <h3 className="text-2xl font-bold">{week.title}</h3>
                        <p className="text-gray-400 mt-1">{week.focus}</p>
                      </div>
                    </div>
                    <div className="text-3xl text-gray-400">
                      {isExpanded ? '−' : '+'}
                    </div>
                  </div>
                </div>

                {isExpanded && (
                  <div className="border-t border-white/10 p-6 space-y-6">
                    {/* Daily Tasks */}
                    <div>
                      <h4 className="text-lg font-semibold mb-4 flex items-center gap-2">
                        <Clock size={20} className="text-blue-400" />
                        Daily Breakdown
                      </h4>
                      <div className="space-y-4">
                        {week.dailyTasks.map((daily, didx) => (
                          <div key={didx} className="bg-white/5 rounded-lg p-4">
                            <div className="font-semibold text-purple-400 mb-2">{daily.day}</div>
                            <ul className="space-y-2">
                              {daily.tasks.map((task, tidx) => (
                                <li key={tidx} className="flex items-start gap-2 text-sm text-gray-300">
                                  <span className="text-green-400 mt-1">→</span>
                                  <span>{task}</span>
                                </li>
                              ))}
                            </ul>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Resources */}
                    <div>
                      <h4 className="text-lg font-semibold mb-4 flex items-center gap-2">
                        <BookOpen size={20} className="text-green-400" />
                        Learning Resources
                      </h4>
                      <div className="grid md:grid-cols-2 gap-3">
                        {week.resources.map((resource, ridx) => (
                          <div key={ridx} className="bg-white/5 rounded-lg p-3 text-sm text-gray-300">
                            • {resource}
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Deliverables */}
                    <div>
                      <h4 className="text-lg font-semibold mb-4 flex items-center gap-2">
                        <Code size={20} className="text-orange-400" />
                        Project Deliverables
                      </h4>
                      <div className="grid md:grid-cols-2 gap-3">
                        {week.deliverables.map((deliverable, didx) => (
                          <div key={didx} className="bg-gradient-to-r from-purple-500/10 to-blue-500/10 rounded-lg p-3 text-sm border border-purple-500/20">
                            <code className="text-purple-300">{deliverable}</code>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {/* Success Metrics */}
        <div className="mt-12 bg-gradient-to-r from-green-500/10 to-blue-500/10 rounded-2xl p-8 border border-green-500/20">
          <h2 className="text-2xl font-bold mb-6">Phase 2 Completion Criteria</h2>
          <div className="grid md:grid-cols-3 gap-6">
            <div className="text-center">
              <div className="text-4xl font-bold text-green-400 mb-2">15+</div>
              <div className="text-gray-300">AI Projects Built</div>
            </div>
            <div className="text-center">
              <div className="text-4xl font-bold text-blue-400 mb-2">3</div>
              <div className="text-gray-300">Production APIs Deployed</div>
            </div>
            <div className="text-center">
              <div className="text-4xl font-bold text-purple-400 mb-2">100%</div>
              <div className="text-gray-300">Portfolio Ready</div>
            </div>
          </div>
        </div>

        {/* Call to Action */}
        <div className="mt-8 text-center">
          <div className="bg-white/5 backdrop-blur-lg rounded-2xl p-8 border border-white/10">
            <h3 className="text-2xl font-bold mb-4">Ready to Start Day 1?</h3>
            <p className="text-gray-300 mb-6">
              Complete the setup checklist above, then dive into Week 1, Day 1. Track your progress by updating your GitHub repo daily.
            </p>
            <div className="inline-block bg-gradient-to-r from-blue-500 to-purple-500 px-8 py-3 rounded-full font-semibold">
              🚀 Let's Build AI!
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}