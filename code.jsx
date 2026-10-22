// App.js
import React, { useState, useEffect, useRef } from 'react';
import Chart from 'chart.js/auto';

function Chatbot() {
  const [userId] = useState(() => 'user_' + Math.floor(Math.random() * 100000));
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([]);

  const sendMessage = async () => {
    if (!input.trim()) return;
    const userMessage = { sender: 'user', text: input };
    setMessages(prev => [...prev, userMessage]);

    try {
      const res = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: userId, message: input })
      });
      const data = await res.json();
      const botMessage = { sender: 'bot', text: data.reply, stress_probability: data.stress_probability };
      setMessages(prev => [...prev, botMessage]);
    } catch {
      setMessages(prev => [...prev, { sender: 'bot', text: "Sorry, I couldn't respond right now." }]);
    }
    setInput('');
  };

  return (
    <div style={{border: '1px solid #ccc', padding: 10, width: 400}}>
      <h3>Mental Health Chatbot</h3>
      <div style={{height: 300, overflowY: 'auto', border: '1px solid #eee', padding: 5, marginBottom: 10}}>
        {messages.map((m, i) => (
          <div key={i} style={{textAlign: m.sender === 'user' ? 'right' : 'left', marginBottom: 6}}>
            <b>{m.sender === 'user' ? 'You' : 'Bot'}:</b> {m.text}
            {m.stress_probability !== undefined && <small> (Stress-language model signal:  {m.stress_probability})</small>}
          </div>
        ))}
      </div>
      <input
        value={input}
        onChange={e => setInput(e.target.value)}
        onKeyDown={e => e.key === 'Enter' && sendMessage()}
        placeholder="Type your message..."
        style={{width: '80%', marginRight: 5}}
      />
      <button onClick={sendMessage}>Send</button>
    </div>
  );
}

function AdminDashboard() {
  const [stats, setStats] = useState(null);
  const chartRef = useRef(null);
  const chartInstance = useRef(null);

  useEffect(() => {
    fetch('/admin/stats')
      .then(res => res.json())
      .then(data => setStats(data));
  }, []);

  useEffect(() => {
    if (stats && chartRef.current) {
      if (chartInstance.current) {
        chartInstance.current.destroy();
      }
      chartInstance.current = new Chart(chartRef.current, {
        type: 'line',
        data: {
          labels: stats.stress_trend.dates,
          datasets: [{
            label: 'Avg Stress-Language Model Signal (%)',
            data: stats.stress_trend.avg_scores,
            borderColor: 'red',
            fill: false,
            tension: 0.3
          }]
        },
        options: {
          scales: {
            y: { beginAtZero: true, max: 100 }
          }
        }
      });
    }
  }, [stats]);

  if (!stats) return <div>Loading admin dashboard...</div>;

  return (
    <div style={{border: '1px solid #ccc', padding: 10, width: 600, marginTop: 20}}>
      <h3>Admin Dashboard</h3>
      <p><b>Active Users (last hour):</b> {stats.total_active_users}</p>
      <canvas ref={chartRef} width="600" height="300"></canvas>
    </div>
  );
}

export default function App() {
  return (
    <div style={{display: 'flex', gap: 20, padding: 20}}>
      <Chatbot />
      <AdminDashboard />
    </div>
  );
}
