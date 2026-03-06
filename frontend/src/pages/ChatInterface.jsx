import React, { useState, useRef, useEffect } from 'react';
import { Send, User, Bot, Loader2, Sparkles, AlertTriangle, Server, Activity, CheckCircle2 } from 'lucide-react';
import { api } from '../api/client';

export default function ChatInterface() {
    const [messages, setMessages] = useState([
        { role: 'assistant', content: 'Hi! I am your Multimodal Insurance Assistant. Ask me anything about your policies, claims, or coverage constraints. I can answer in English, Hindi, Tamil, and more!' }
    ]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const [language, setLanguage] = useState('english');
    const messagesEndRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const handleSend = async (e) => {
        e.preventDefault();
        if (!input.trim()) return;

        const userMessage = { role: 'user', content: input };
        setMessages(prev => [...prev, userMessage]);
        setInput('');
        setLoading(true);

        try {
            const response = await api.queryAssistant({
                query: userMessage.content,
                language
            });

            setMessages(prev => [...prev, { role: 'assistant', content: response.answer, meta: response }]);
        } catch (err) {
            setMessages(prev => [...prev, { role: 'assistant', content: '🚨 ' + (err.response?.data?.detail || err.message || 'Error reaching the assistant.') }]);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="fade-in" style={{ display: 'flex', flexDirection: 'column', height: '100%', maxWidth: '1000px', margin: '0 auto' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', marginBottom: '24px' }}>
                <div>
                    <h1 className="gradient-text" style={{ fontSize: '2.5rem', marginBottom: '8px' }}>Insurance Assistant</h1>
                    <p style={{ color: 'var(--text-secondary)' }}>Enterprise Intelligence & Document Search</p>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{ fontSize: '0.9rem', color: 'var(--text-muted)' }}>Response Language:</span>
                    <select
                        value={language}
                        onChange={(e) => setLanguage(e.target.value)}
                        style={{ padding: '8px 12px', borderRadius: '8px', border: '1px solid var(--glass-border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)' }}
                    >
                        <option value="english">English</option>
                        <option value="hindi">Hindi (हिन्दी)</option>
                        <option value="tamil">Tamil (தமிழ்)</option>
                        <option value="telugu">Telugu (తెలుగు)</option>
                        <option value="marathi">Marathi (मराठी)</option>
                    </select>
                </div>
            </div>

            <div className="glass-panel" style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', padding: '0' }}>
                <div style={{ flex: 1, overflowY: 'auto', padding: '24px', display: 'flex', flexDirection: 'column', gap: '24px' }}>
                    {messages.map((msg, idx) => (
                        <div key={idx} style={{
                            display: 'flex',
                            gap: '16px',
                            alignItems: 'flex-start',
                            flexDirection: msg.role === 'user' ? 'row-reverse' : 'row'
                        }}>
                            <div style={{
                                width: '40px', height: '40px', borderRadius: '50%', flexShrink: 0,
                                display: 'flex', alignItems: 'center', justifyContent: 'center',
                                background: msg.role === 'user' ? 'var(--bg-tertiary)' : 'var(--accent-glow)',
                                border: `1px solid ${msg.role === 'user' ? 'var(--glass-border)' : 'var(--accent-blue)'}`
                            }}>
                                {msg.role === 'user' ? <User size={20} /> : <Bot size={20} color="var(--accent-blue)" />}
                            </div>

                            <div style={{
                                maxWidth: '75%',
                                padding: '16px',
                                borderRadius: '16px',
                                background: msg.role === 'user' ? 'var(--accent-gradient)' : 'var(--bg-secondary)',
                                border: msg.role === 'user' ? 'none' : '1px solid var(--glass-border)',
                                whiteSpace: 'pre-wrap',
                                lineHeight: '1.6',
                                color: msg.role === 'user' ? 'white' : 'var(--text-primary)',
                                borderTopRightRadius: msg.role === 'user' ? '4px' : '16px',
                                borderTopLeftRadius: msg.role === 'user' ? '16px' : '4px',
                            }}>
                                {msg.content}

                                {msg.meta && (msg.meta.retrieved_count > 0) && (
                                    {/* Metadata and chunk info removed for cleaner UI */ }
                                )}
                            </div>
                        </div>
                    ))}
                    {loading && (
                        <div style={{ display: 'flex', gap: '16px', alignItems: 'flex-start' }}>
                            <div style={{ width: '40px', height: '40px', borderRadius: '50%', background: 'var(--accent-glow)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                                <Loader2 className="animate-spin" size={20} color="var(--accent-blue)" />
                            </div>
                            <div style={{ padding: '16px', borderRadius: '16px', background: 'var(--bg-secondary)', border: '1px solid var(--glass-border)', color: 'var(--text-muted)' }}>
                                Analyzing policy documents...
                            </div>
                        </div>
                    )}
                    <div ref={messagesEndRef} />
                </div>

                <form onSubmit={handleSend} style={{ padding: '24px', borderTop: '1px solid var(--glass-border)', background: 'var(--bg-color)', display: 'flex', gap: '16px' }}>
                    <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        disabled={loading}
                        placeholder="E.g., Does my policy cover pre-existing knee surgery?"
                        style={{
                            flex: 1,
                            padding: '16px',
                            borderRadius: '12px',
                            border: '1px solid var(--glass-border)',
                            background: 'var(--bg-secondary)',
                            color: 'var(--text-primary)',
                            fontSize: '1rem'
                        }}
                    />
                    <button
                        type="submit"
                        disabled={loading || !input.trim()}
                        className="btn-primary"
                        style={{ padding: '0 24px', borderRadius: '12px', opacity: (loading || !input.trim()) ? 0.5 : 1 }}
                    >
                        <Send size={20} />
                    </button>
                </form>
            </div>
        </div>
    );
}
