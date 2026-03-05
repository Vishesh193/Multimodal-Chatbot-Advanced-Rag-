import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { Bot, GitCompare, FileText, FileSearch, ArrowRight, ChevronDown, ChevronUp, Mail, Send } from 'lucide-react';

export default function LandingPage() {
    const [openFaq, setOpenFaq] = useState(null);

    const toggleFaq = (index) => {
        setOpenFaq(openFaq === index ? null : index);
    };

    const faqs = [
        {
            q: "Does my policy cover knee surgery?",
            a: "Our AI Insurance Assistant reads your exact policy document. You can simply ask 'Does my policy cover knee surgery?' and it will extract coverage details, waiting periods, and exclusions directly from your specific PDF."
        },
        {
            q: "How do I file a health insurance claim?",
            a: "Use our 'Claim Guide' feature. Describe your hospital visit, and the AI will generate a personalized step-by-step checklist of the documents and actions you need to take according to your insurer's rules."
        },
        {
            q: "What are pre-existing condition exclusions?",
            a: "Insurers often have waiting periods for conditions you had before buying the policy. Our system highlights these exclusions clearly, so you're not caught off-guard during a medical emergency."
        },
        {
            q: "How long does claim approval take?",
            a: "While processing times vary by insurer (typically 15-30 days), our system will tell you the exact timeline specified in your policy booklet for reimbursement or cashless claim settlements."
        }
    ];

    return (
        <div style={{ backgroundColor: 'white', color: 'var(--text-primary)' }}>
            {/* ── 1. HERO SECTION ── */}
            <section id="home" style={{ padding: '120px 5%', display: 'flex', alignItems: 'center', gap: '64px', minHeight: '90vh' }}>
                <div style={{ flex: 1 }}>
                    <div style={{ display: 'inline-block', padding: '8px 16px', background: 'var(--accent-light)', color: 'var(--accent-deep)', borderRadius: '24px', fontWeight: 600, fontSize: '0.9rem', marginBottom: '24px' }}>
                        Introducing InsureAI 2.0
                    </div>
                    <h1 style={{ fontSize: '4.5rem', lineHeight: 1.1, marginBottom: '24px', letterSpacing: '-1px' }}>
                        AI-Powered <br /><span className="gradient-text">Insurance Policy</span> <br />Intelligence
                    </h1>
                    <p style={{ fontSize: '1.25rem', color: 'var(--text-secondary)', marginBottom: '40px', maxWidth: '600px', lineHeight: 1.6 }}>
                        Understand your insurance policies instantly. InsureAI helps you analyze coverage, highlight hidden exclusions, and navigate claim procedures using advanced AI and document retrieval technology.
                    </p>
                    <div style={{ display: 'flex', gap: '16px' }}>
                        <Link to="/app/chat" className="btn-primary" style={{ padding: '16px 32px', fontSize: '1.1rem' }}>
                            Try Insurance Assistant <ArrowRight size={20} />
                        </Link>
                        <Link to="/app/policies" className="btn-secondary" style={{ padding: '16px 32px', fontSize: '1.1rem' }}>
                            Upload Policy
                        </Link>
                    </div>

                    <div style={{ marginTop: '48px', display: 'flex', gap: '32px', alignItems: 'center', color: 'var(--text-muted)', fontSize: '0.95rem' }}>
                        <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>✅ SOC2 Compliant </span>
                        <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>✅ Local Privacy</span>
                        <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>✅ 12+ Languages</span>
                    </div>
                </div>

                <div style={{ flex: 1, position: 'relative', display: 'flex', justifyContent: 'center' }}>
                    <div style={{
                        width: '100%',
                        maxWidth: '600px',
                        position: 'relative',
                        zIndex: 1
                    }}>
                        <img
                            src="/hero-insurance.png"
                            alt="InsureAI Platform"
                            style={{
                                width: '100%',
                                height: 'auto',
                                borderRadius: '24px',
                                boxShadow: '0 20px 40px -12px rgba(0, 0, 0, 0.1)',
                                animation: 'fadeIn 0.8s ease-out'
                            }}
                        />
                        {/* Decorative background blur */}
                        <div style={{
                            position: 'absolute',
                            top: '50%',
                            left: '50%',
                            transform: 'translate(-50%, -50%)',
                            width: '80%',
                            height: '80%',
                            background: 'var(--accent-blue)',
                            opacity: 0.1,
                            borderRadius: '50%',
                            filter: 'blur(80px)',
                            zIndex: -1
                        }}></div>
                    </div>
                </div>
            </section>

            {/* ── 2. FEATURES SECTION ── */}
            <section id="features" style={{ padding: '120px 5%', background: 'var(--bg-secondary)' }}>
                <div style={{ textAlign: 'center', marginBottom: '80px' }}>
                    <h2 style={{ fontSize: '3rem', marginBottom: '16px' }}>Powerful features to demystify insurance</h2>
                    <p style={{ fontSize: '1.25rem', color: 'var(--text-secondary)', maxWidth: '600px', margin: '0 auto' }}>
                        Our complete suite of RAG-powered tools ensures you never get lost in a 100-page policy document again.
                    </p>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '32px' }}>

                    <div className="glass-panel" style={{ padding: '40px 32px', background: 'white' }}>
                        <div style={{ width: '64px', height: '64px', background: 'var(--accent-light)', borderRadius: '16px', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '24px' }}>
                            <Bot size={32} color="var(--accent-blue)" />
                        </div>
                        <h3 style={{ fontSize: '1.5rem', marginBottom: '12px' }}>AI Insurance Assistant</h3>
                        <p style={{ color: 'var(--text-secondary)' }}>Chat naturally with your policy documents. Get completely accurate answers about coverage, limits, and claims directly cited from your contract.</p>
                    </div>

                    <div className="glass-panel" style={{ padding: '40px 32px', background: 'white' }}>
                        <div style={{ width: '64px', height: '64px', background: 'var(--accent-light)', borderRadius: '16px', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '24px' }}>
                            <GitCompare size={32} color="var(--accent-blue)" />
                        </div>
                        <h3 style={{ fontSize: '1.5rem', marginBottom: '12px' }}>Policy Comparison</h3>
                        <p style={{ color: 'var(--text-secondary)' }}>Compare multiple insurance policies side-by-side to find the best coverage options and clearly see hidden differences.</p>
                    </div>

                    <div className="glass-panel" style={{ padding: '40px 32px', background: 'white' }}>
                        <div style={{ width: '64px', height: '64px', background: 'var(--accent-light)', borderRadius: '16px', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '24px' }}>
                            <FileText size={32} color="var(--accent-blue)" />
                        </div>
                        <h3 style={{ fontSize: '1.5rem', marginBottom: '12px' }}>Claim Guidance</h3>
                        <p style={{ color: 'var(--text-secondary)' }}>Don't let claims get rejected. Get a personalized, step-by-step checklist of documents needed for your specific hospital claim.</p>
                    </div>

                    <div className="glass-panel" style={{ padding: '40px 32px', background: 'white' }}>
                        <div style={{ width: '64px', height: '64px', background: 'var(--accent-light)', borderRadius: '16px', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '24px' }}>
                            <FileSearch size={32} color="var(--accent-blue)" />
                        </div>
                        <h3 style={{ fontSize: '1.5rem', marginBottom: '12px' }}>Smart Bill Scanner</h3>
                        <p style={{ color: 'var(--text-secondary)' }}>Upload medical bills and discharge summaries. Our vision AI will automatically extract JSON data and analyze claim eligibility.</p>
                    </div>

                </div>
            </section>

            {/* ── 3. FAQ SECTION ── */}
            <section id="faq" style={{ padding: '120px 5%', maxWidth: '900px', margin: '0 auto' }}>
                <h2 style={{ fontSize: '3rem', textAlign: 'center', marginBottom: '16px' }}>Frequently Asked Questions</h2>
                <p style={{ fontSize: '1.25rem', color: 'var(--text-secondary)', textAlign: 'center', marginBottom: '64px' }}>
                    Everything you need to know about navigating your policies using InsureAI.
                </p>

                <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                    {faqs.map((faq, idx) => (
                        <div key={idx} className="glass-panel" style={{ padding: '24px', cursor: 'pointer', background: 'white' }} onClick={() => toggleFaq(idx)}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                <h4 style={{ fontSize: '1.1rem', fontWeight: 500, margin: 0 }}>{faq.q}</h4>
                                {openFaq === idx ? <ChevronUp size={20} color="var(--accent-deep)" /> : <ChevronDown size={20} color="var(--text-muted)" />}
                            </div>
                            {openFaq === idx && (
                                <p style={{ marginTop: '16px', color: 'var(--text-secondary)', lineHeight: 1.6, borderTop: '1px solid var(--glass-border)', paddingTop: '16px' }}>
                                    {faq.a}
                                </p>
                            )}
                        </div>
                    ))}
                </div>
            </section>

            {/* ── 4. CONTACT SECTION ── */}
            <section id="contact" style={{ padding: '120px 5%', background: 'var(--accent-deep)', color: 'white' }}>
                <div style={{ maxWidth: '1000px', margin: '0 auto', display: 'flex', flexWrap: 'wrap', gap: '64px' }}>

                    <div style={{ flex: 1, minWidth: '300px' }}>
                        <h2 style={{ fontSize: '3rem', marginBottom: '24px', color: 'white' }}>Get in touch</h2>
                        <p style={{ fontSize: '1.1rem', opacity: 0.9, marginBottom: '40px', lineHeight: 1.6 }}>
                            Have questions about integrating InsureAI into your brokerage or agency? Our team is here to help you deploy smarter insurance operations.
                        </p>

                        <div style={{ display: 'flex', alignItems: 'center', gap: '16px', marginBottom: '24px', fontSize: '1.1rem' }}>
                            <div style={{ width: '48px', height: '48px', borderRadius: '50%', background: 'rgba(255,255,255,0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                                <Mail size={24} color="white" />
                            </div>
                            <a href="mailto:support@insureai.com" style={{ color: 'white', textDecoration: 'none', fontWeight: 500 }}>support@insureai.com</a>
                        </div>

                        <p style={{ opacity: 0.8, fontSize: '0.9rem', marginTop: '32px' }}>
                            * Our support team will respond within 24 hours.
                        </p>
                    </div>

                    <div className="glass-panel" style={{ flex: 1.2, minWidth: '350px', background: 'white', padding: '40px', color: 'var(--text-primary)' }}>
                        <form onSubmit={(e) => e.preventDefault()} style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                            <div>
                                <label style={{ display: 'block', marginBottom: '8px', fontSize: '0.9rem', fontWeight: 500 }}>Full Name</label>
                                <input type="text" placeholder="John Doe" style={{ width: '100%', padding: '12px 16px', borderRadius: '8px', border: '1px solid var(--glass-border)', fontSize: '1rem' }} />
                            </div>
                            <div>
                                <label style={{ display: 'block', marginBottom: '8px', fontSize: '0.9rem', fontWeight: 500 }}>Email Address</label>
                                <input type="email" placeholder="john@example.com" style={{ width: '100%', padding: '12px 16px', borderRadius: '8px', border: '1px solid var(--glass-border)', fontSize: '1rem' }} />
                            </div>
                            <div>
                                <label style={{ display: 'block', marginBottom: '8px', fontSize: '0.9rem', fontWeight: 500 }}>Message</label>
                                <textarea placeholder="How can we help you?" rows="4" style={{ width: '100%', padding: '12px 16px', borderRadius: '8px', border: '1px solid var(--glass-border)', fontSize: '1rem', resize: 'vertical' }}></textarea>
                            </div>
                            <button className="btn-primary" style={{ marginTop: '16px', justifyContent: 'center', padding: '16px' }}>
                                Send Query <Send size={18} />
                            </button>
                        </form>
                    </div>

                </div>
            </section>

        </div>
    );
}
