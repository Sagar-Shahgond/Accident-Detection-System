import { useState, useEffect } from 'react'
import { supabase } from './supabaseClient'
import './index.css'

const severityColors = {
  MINOR: 'bg-green-100 text-green-800 border-green-300',
  MAJOR: 'bg-orange-100 text-orange-800 border-orange-300',
  CRITICAL: 'bg-red-100 text-red-800 border-red-300 animate-pulse',
}

function App() {
  const [incidents, setIncidents] = useState([])
  const [loading, setLoading] = useState(true)

  // Load existing incidents on first render
  useEffect(() => {
    async function fetchIncidents() {
      const { data, error } = await supabase
        .from('incidents')
        .select('*')
        .order('created_at', { ascending: false })

      if (error) {
        console.error('Error fetching incidents:', error)
      } else {
        setIncidents(data)
      }
      setLoading(false)
    }

    fetchIncidents()
  }, [])

  // Subscribe to realtime updates - new incidents appear instantly
  useEffect(() => {
    const channel = supabase
      .channel('incidents-realtime')
      .on(
        'postgres_changes',
        { event: 'INSERT', schema: 'public', table: 'incidents' },
        (payload) => {
          console.log('New incident received:', payload.new)
          setIncidents((prev) => [payload.new, ...prev])
        }
      )
      .subscribe()

    return () => {
      supabase.removeChannel(channel)
    }
  }, [])

  return (
    <div className="min-h-screen bg-gray-100 p-6">
      <header className="mb-6">
        <h1 className="text-3xl font-bold text-gray-900">TransitGuard AI</h1>
        <p className="text-gray-500">Live Incident Dashboard</p>
      </header>

      {loading ? (
        <p className="text-gray-500">Loading incidents...</p>
      ) : incidents.length === 0 ? (
        <p className="text-gray-500">No incidents detected yet.</p>
      ) : (
        <div className="space-y-4">
          {incidents.map((incident) => (
            <div
              key={incident.id}
              className="bg-white rounded-lg shadow p-4 border"
            >
              <div className="flex items-center justify-between mb-2">
                <span
                  className={`px-3 py-1 rounded-full text-sm font-semibold border ${
                    severityColors[incident.severity_level] || 'bg-gray-100'
                  }`}
                >
                  {incident.severity_level} ({incident.severity_score}/100)
                </span>
                <span className="text-sm text-gray-400">
                  {new Date(incident.created_at).toLocaleString()}
                </span>
              </div>

              <h2 className="text-lg font-semibold text-gray-800">
                {incident.accident_type.replace('-', ' ').toUpperCase()}
              </h2>
              <p className="text-gray-600 text-sm mb-2">{incident.location}</p>
              <p className="text-gray-700">{incident.incident_summary}</p>

              <div className="mt-3 flex gap-2">
                {incident.notify?.map((authority) => (
                  <span
                    key={authority}
                    className="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded-full"
                  >
                    {authority}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default App