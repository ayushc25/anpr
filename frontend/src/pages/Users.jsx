import { useEffect, useState } from 'react'
import client from '../api/client'
import Layout from '../components/Layout'

const EMPTY = { username: '', full_name: '', password: '', role_name: '', permissions: [], is_active: true }

export default function Users() {
  const [users, setUsers] = useState([])
  const [permissions, setPermissions] = useState([])
  const [form, setForm] = useState(EMPTY)
  const [editingId, setEditingId] = useState(null)
  const [error, setError] = useState('')

  async function load() {
    const [usersRes, permsRes] = await Promise.all([
      client.get('/users'),
      client.get('/users/permissions'),
    ])
    setUsers(usersRes.data)
    setPermissions(permsRes.data)
  }

  useEffect(() => { load() }, [])

  function togglePermission(key) {
    setForm((f) => ({
      ...f,
      permissions: f.permissions.includes(key)
        ? f.permissions.filter((p) => p !== key)
        : [...f.permissions, key],
    }))
  }

  async function submit(e) {
    e.preventDefault()
    setError('')
    try {
      if (editingId) {
        const { username, ...rest } = form
        await client.put(`/users/${editingId}`, rest)
      } else {
        await client.post('/users', form)
      }
      setForm(EMPTY)
      setEditingId(null)
      load()
    } catch (err) {
      setError(err?.response?.data?.detail || 'Failed to save user')
    }
  }

  function edit(u) {
    setEditingId(u.id)
    setForm({
      username: u.username,
      full_name: u.full_name,
      password: '',
      role_name: u.role_name,
      permissions: u.permissions || [],
      is_active: u.is_active,
    })
  }

  async function remove(id) {
    if (!confirm('Delete this user?')) return
    await client.delete(`/users/${id}`)
    load()
  }

  return (
    <Layout title="Users">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <form onSubmit={submit} className="card p-4 space-y-3">
          <div className="font-semibold text-app-primary mb-1">{editingId ? 'Edit User' : 'Add User'}</div>
          {error && <div className="bg-red-500/10 text-red-400 text-xs rounded px-2 py-1">{error}</div>}
          <div>
            <label className="block text-xs text-app-secondary mb-1">Username</label>
            <input
              value={form.username}
              disabled={!!editingId}
              onChange={(e) => setForm((f) => ({ ...f, username: e.target.value }))}
              className="w-full px-3 py-2 rounded-lg bg-app border border-app text-app-primary text-sm disabled:opacity-50"
            />
          </div>
          <div>
            <label className="block text-xs text-app-secondary mb-1">Full Name</label>
            <input
              value={form.full_name}
              onChange={(e) => setForm((f) => ({ ...f, full_name: e.target.value }))}
              className="w-full px-3 py-2 rounded-lg bg-app border border-app text-app-primary text-sm"
            />
          </div>
          <div>
            <label className="block text-xs text-app-secondary mb-1">{editingId ? 'New Password (optional)' : 'Password'}</label>
            <input
              type="password"
              value={form.password}
              onChange={(e) => setForm((f) => ({ ...f, password: e.target.value }))}
              className="w-full px-3 py-2 rounded-lg bg-app border border-app text-app-primary text-sm"
            />
          </div>
          <div>
            <label className="block text-xs text-app-secondary mb-1">Role Name</label>
            <input
              value={form.role_name}
              onChange={(e) => setForm((f) => ({ ...f, role_name: e.target.value }))}
              placeholder="e.g. Site Guard, Shift Supervisor"
              className="w-full px-3 py-2 rounded-lg bg-app border border-app text-app-primary text-sm"
            />
          </div>
          <div>
            <label className="block text-xs text-app-secondary mb-1.5">Visible Menus</label>
            <div className="grid grid-cols-2 gap-1.5">
              {permissions.map((p) => (
                <label key={p.key} className="flex items-center gap-2 text-sm text-app-secondary">
                  <input
                    type="checkbox"
                    checked={form.permissions.includes(p.key)}
                    onChange={() => togglePermission(p.key)}
                  />
                  {p.label}
                </label>
              ))}
            </div>
          </div>
          <label className="flex items-center gap-2 text-sm text-app-secondary">
            <input type="checkbox" checked={form.is_active} onChange={(e) => setForm((f) => ({ ...f, is_active: e.target.checked }))} />
            Active
          </label>
          <div className="flex gap-2">
            <button className="flex-1 py-2 rounded-lg bg-blue-600 hover:bg-blue-700 text-app-primary text-sm">
              {editingId ? 'Update' : 'Add'}
            </button>
            {editingId && (
              <button type="button" onClick={() => { setEditingId(null); setForm(EMPTY) }} className="px-3 py-2 rounded-lg surface-1 text-app-secondary text-sm">
                Cancel
              </button>
            )}
          </div>
        </form>

        <div className="card p-4 lg:col-span-2">
          <div className="font-semibold text-app-primary mb-3">Users</div>
          <div className="overflow-x-auto">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Username</th>
                  <th>Full Name</th>
                  <th>Role</th>
                  <th>Menus</th>
                  <th>Status</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {users.map((u) => (
                  <tr key={u.id}>
                    <td>{u.username}</td>
                    <td>{u.full_name}</td>
                    <td>{u.role_name}</td>
                    <td className="text-xs">{(u.permissions || []).length} of {permissions.length}</td>
                    <td>
                      <span className={`text-xs font-semibold ${u.is_active ? 'text-emerald-400' : 'text-red-400'}`}>
                        {u.is_active ? 'Active' : 'Disabled'}
                      </span>
                    </td>
                    <td className="text-right space-x-2 whitespace-nowrap">
                      <button onClick={() => edit(u)} className="text-blue-400 hover:underline text-xs">Edit</button>
                      <button onClick={() => remove(u.id)} className="text-red-400 hover:underline text-xs">Delete</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </Layout>
  )
}
