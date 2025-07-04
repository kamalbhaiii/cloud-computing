import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Dashboard from "./pages/dashboard/index";
import NotFound from "./pages/notfound/index";
import Navbar from "./components/navbar/index";
import Database from "./pages/database";

export default function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-100 dark:bg-gray-950 transition-colors">
        <Navbar />
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/database" element={<Database />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </div>
    </Router>
  );
}
