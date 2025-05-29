import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Dashboard from "./pages/dashboard/index";
import Navbar from "./components/navbar/index";

export default function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-100 dark:bg-gray-950 transition-colors">
        <Navbar />
        <Routes>
          <Route path="/" element={<Dashboard />} />
          {/* Add more routes if needed */}
        </Routes>
      </div>
    </Router>
  );
}
