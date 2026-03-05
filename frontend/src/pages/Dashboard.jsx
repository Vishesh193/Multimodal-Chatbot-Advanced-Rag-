import React, { useEffect, useState } from 'react';
import { api } from '../api/client';
import { ShieldAlert, CheckCircle2, Server, Database, Activity } from 'lucide-react';

export default function Dashboard() {
    const [status, setStatus] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        api.getStatus()
            .then(res => setStatus(res))
            .catch(err => setError(err.message))
            .finally(() => setLoading(false));
    }, []);

    if (loading) return <div className="p-8 text-center text-secondary">Loading system status...</div>;
    if (error) return <div className="p-8 text-center text-error"><h3>Server Offline</h3><p>Make sure FastAPI is running on port 8000.</p></div>;

    return (
        <div className="fade-in">
            <h1 className="gradient-text" style={{ fontSize: '2.5rem', marginBottom: '8px' }}>
                System Overview
            </h1>
            <p style={{ color: 'var(--text-secondary)', marginBottom: '32px' }}>
                Real-time health of the Multimodal Insurance RAG pipeline
            </p>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '24px' }}>

                {/* Memory Stats */}
                <div style={{ background: 'var(--bg-secondary)', padding: '24px', borderRadius: '12px', border: '1px solid var(--glass-border)' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                        <h3 style={{ fontSize: '1.2rem', fontWeight: 500 }}>Vector Store</h3>
                        <Database color="var(--accent-blue)" />
                    </div>
                    <div style={{ fontSize: '2.5rem', fontWeight: 700, color: 'white', marginBottom: '8px' }}>
                        {status?.vector_store.total_documents || 0}
                    </div>
                    <p style={{ color: 'var(--text-muted)' }}>Total embedded document chunks</p>
                    <div style={{ marginTop: '16px', fontSize: '0.85rem', color: 'var(--success)', display: 'flex', alignItems: 'center', gap: '4px' }}>
                        <CheckCircle2 size={14} /> ChromaDB Connected
                    </div>
                </div>

                {/* Policies */}
                <div style={{ background: 'var(--bg-secondary)', padding: '24px', borderRadius: '12px', border: '1px solid var(--glass-border)' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                        <h3 style={{ fontSize: '1.2rem', fontWeight: 500 }}>Active Policies</h3>
                        <ShieldAlert color="var(--warning)" />
                    </div>
                    <div style={{ fontSize: '2.5rem', fontWeight: 700, color: 'white', marginBottom: '8px' }}>
                        {status?.insurance?.registered_policies || 0}
                    </div>
                    <p style={{ color: 'var(--text-muted)' }}>Ingested insurance policies</p>
                </div>

                {/* LLM Routing */}
                <div style={{ background: 'var(--bg-secondary)', padding: '24px', borderRadius: '12px', border: '1px solid var(--glass-border)' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                        <h3 style={{ fontSize: '1.2rem', fontWeight: 500 }}>LLM Router</h3>
                        <Server color="var(--accent-purple)" />
                    </div>
                    <p style={{ color: 'var(--text-secondary)', margin: '8px 0', display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <Activity size={16} color={status?.llm_router.groq_available ? 'var(--success)' : 'var(--error)'} />
                        Groq API (Text)
                    </p>
                    <p style={{ color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <Activity size={16} color={status?.llm_router.ollama_available ? 'var(--success)' : 'var(--warning)'} />
                        Ollama (Vision)
                    </p>
                </div>

            </div>

            {status?.insurance?.supported_languages && (
                <div style={{ marginTop: '32px', padding: '24px', background: 'var(--bg-secondary)', borderRadius: '12px', border: '1px solid var(--glass-border)' }}>
                    <h3 style={{ marginBottom: '16px' }}>Multilingual Support Active</h3>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                        {status.insurance.supported_languages.map(lang => (
                            <span key={lang} style={{ padding: '6px 12px', background: 'var(--bg-color)', border: '1px solid var(--glass-border)', borderRadius: '16px', fontSize: '0.85rem', textTransform: 'capitalize' }}>
                                {lang}
                            </span>
                        ))}
                    </div>
                </div>
            )}

        </div>
    );
}
