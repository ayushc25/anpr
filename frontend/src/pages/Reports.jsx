import { useEffect, useState } from 'react'
import client from '../api/client'
import Layout from '../components/Layout'

const PAGE_SIZE = 10

function todayStr() {
  return new Date().toISOString().slice(0, 10)
}

async function downloadBlob(params, filename) {
  const res = await client.get('/reports/export', { params, responseType: 'blob' })
  const url = window.URL.createObjectURL(new Blob([res.data]))
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
  window.URL.revokeObjectURL(url)
}

export default function Reports() {
  const [filters, setFilters] = useState({ date_from: '', date_to: '', status: '', vehicle_type: '' })
  const [loading, setLoading] = useState(false)

  const [dailyRange, setDailyRange] = useState({ date_from: '', date_to: '' })
  const [dailyRows, setDailyRows] = useState([])
  const [dailyTotal, setDailyTotal] = useState(0)
  const [dailyPage, setDailyPage] = useState(1)
  const [dailyLoading, setDailyLoading] = useState(false)
  const [dailyDownloadingDate, setDailyDownloadingDate] = useState(null)

  function setToday() {
    const t = todayStr()
    setFilters((f) => ({ ...f, date_from: t, date_to: t }))
  }

  async function download() {
    setLoading(true)
    try {
      const params = { report_type: 'detailed' }
      if (filters.date_from) params.date_from = filters.date_from
      if (filters.date_to) params.date_to = filters.date_to
      if (filters.status) params.status = filters.status
      if (filters.vehicle_type) params.vehicle_type = filters.vehicle_type
      await downloadBlob(params, 'anpr_report.xlsx')
    } finally {
      setLoading(false)
    }
  }

  async function loadDailySummary(page) {
    setDailyLoading(true)
    try {
      const params = { page, page_size: PAGE_SIZE }
      if (dailyRange.date_from) params.date_from = dailyRange.date_from
      if (dailyRange.date_to) params.date_to = `${dailyRange.date_to}T23:59:59`
      const res = await client.get('/reports/daily-summary', { params })
      setDailyRows(res.data.items)
      setDailyTotal(res.data.total)
      setDailyPage(page)
    } finally {
      setDailyLoading(false)
    }
  }

  useEffect(() => { loadDailySummary(1) }, [dailyRange.date_from, dailyRange.date_to])

  async function downloadDay(date) {
    setDailyDownloadingDate(date)
    try {
      await downloadBlob(
        { report_type: 'daily', date_from: date, date_to: `${date}T23:59:59` },
        `anpr_daily_summary_${date}.xlsx`,
      )
    } finally {
      setDailyDownloadingDate(null)
    }
  }

  const dailyTotalPages = Math.max(1, Math.ceil(dailyTotal / PAGE_SIZE))

  return (
    <Layout title="Reports">
      <div className="card p-6 max-w-xl">
        <div className="font-semibold text-app-primary mb-4">Export ANPR Report (Excel)</div>

        <div className="flex items-center justify-between mb-1">
          <label className="block text-xs text-app-secondary">Date Range</label>
          <button type="button" onClick={setToday} className="text-xs text-blue-400 hover:underline">
            Today only
          </button>
        </div>
        <div className="grid grid-cols-2 gap-4 mb-4">
          <div>
            <input
              type="date"
              value={filters.date_from}
              onChange={(e) => setFilters((f) => ({ ...f, date_from: e.target.value }))}
              className="w-full px-3 py-2 rounded-lg bg-app border border-app text-app-primary text-sm"
            />
          </div>
          <div>
            <input
              type="date"
              value={filters.date_to}
              onChange={(e) => setFilters((f) => ({ ...f, date_to: e.target.value }))}
              className="w-full px-3 py-2 rounded-lg bg-app border border-app text-app-primary text-sm"
            />
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4 mb-6">
          <div>
            <label className="block text-xs text-app-secondary mb-1">Vehicle Status</label>
            <select
              value={filters.status}
              onChange={(e) => setFilters((f) => ({ ...f, status: e.target.value }))}
              className="w-full px-3 py-2 rounded-lg bg-app border border-app text-app-primary text-sm"
            >
              <option value="">All</option>
              <option value="registered">Registered</option>
              <option value="whitelist">Whitelist</option>
              <option value="blacklist">Blacklist</option>
              <option value="unknown">Unknown</option>
            </select>
          </div>
          <div>
            <label className="block text-xs text-app-secondary mb-1">Vehicle Type</label>
            <select
              value={filters.vehicle_type}
              onChange={(e) => setFilters((f) => ({ ...f, vehicle_type: e.target.value }))}
              className="w-full px-3 py-2 rounded-lg bg-app border border-app text-app-primary text-sm"
            >
              <option value="">All</option>
              <option value="car">Car</option>
              <option value="motorbike">Motorbike</option>
              <option value="bicycle">Bicycle</option>
              <option value="truck">Truck</option>
              <option value="bus">Bus</option>
            </select>
          </div>
        </div>

        <button
          onClick={download}
          disabled={loading}
          className="px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-700 text-app-primary text-sm disabled:opacity-50"
        >
          {loading ? 'Generating…' : 'Download Excel Report'}
        </button>
      </div>

      <div className="card p-6 max-w-3xl mt-4">
        <div className="flex items-center justify-between mb-4">
          <div className="font-semibold text-app-primary">Daily Summary</div>
          <div className="flex items-center gap-2">
            <input
              type="date"
              value={dailyRange.date_from}
              onChange={(e) => setDailyRange((f) => ({ ...f, date_from: e.target.value }))}
              className="px-3 py-1.5 rounded-lg bg-app border border-app text-app-primary text-sm"
            />
            <span className="text-app-muted text-sm">to</span>
            <input
              type="date"
              value={dailyRange.date_to}
              onChange={(e) => setDailyRange((f) => ({ ...f, date_to: e.target.value }))}
              className="px-3 py-1.5 rounded-lg bg-app border border-app text-app-primary text-sm"
            />
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="data-table">
            <thead>
              <tr>
                <th>Date</th>
                <th>Total</th>
                <th>Entries</th>
                <th>Exits</th>
                <th>Registered</th>
                <th>Whitelist</th>
                <th>Blacklist</th>
                <th>Unknown</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {!dailyLoading && dailyRows.length === 0 && (
                <tr>
                  <td colSpan={9} className="text-center text-app-muted py-6">No data for this range</td>
                </tr>
              )}
              {dailyRows.map((row) => (
                <tr key={row.date}>
                  <td>{row.date}</td>
                  <td>{row.total}</td>
                  <td>{row.entries}</td>
                  <td>{row.exits}</td>
                  <td>{row.registered}</td>
                  <td>{row.whitelist}</td>
                  <td>{row.blacklist}</td>
                  <td>{row.unknown}</td>
                  <td className="text-right">
                    <button
                      onClick={() => downloadDay(row.date)}
                      disabled={dailyDownloadingDate === row.date}
                      className="text-blue-400 hover:underline text-xs disabled:opacity-50"
                    >
                      {dailyDownloadingDate === row.date ? 'Downloading…' : 'Download'}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="flex items-center justify-between mt-4 text-sm text-app-secondary">
          <span>Page {dailyPage} of {dailyTotalPages}</span>
          <div className="flex gap-2">
            <button
              type="button"
              onClick={() => loadDailySummary(dailyPage - 1)}
              disabled={dailyPage <= 1 || dailyLoading}
              className="px-3 py-1.5 rounded-lg surface-1 hover-surface-2 disabled:opacity-40"
            >
              Prev
            </button>
            <button
              type="button"
              onClick={() => loadDailySummary(dailyPage + 1)}
              disabled={dailyPage >= dailyTotalPages || dailyLoading}
              className="px-3 py-1.5 rounded-lg surface-1 hover-surface-2 disabled:opacity-40"
            >
              Next
            </button>
          </div>
        </div>
      </div>
    </Layout>
  )
}
