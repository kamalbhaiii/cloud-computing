import { Link, useLocation } from "react-router-dom";
import { routes } from "../../routes";
import { useState, useEffect } from "react";
import { Menu, X, Sun, Moon } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

export default function Navbar() {
  const location = useLocation();
  const [menuOpen, setMenuOpen] = useState(false);

  const [theme, setTheme] = useState(() =>
    localStorage.getItem("theme") || "light"
  );

  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark");
    localStorage.setItem("theme", theme);
  }, [theme]);

  return (
    <nav className="bg-white dark:bg-gray-900 border-b border-gray-200 dark:border-gray-800 shadow-sm">
      <div className="max-w-7xl mx-auto px-4 py-3 flex items-center justify-between">
        <h1 className="text-xl font-bold text-gray-800 dark:text-white">
          Cloud Computing
        </h1>

        {/* Desktop Menu */}
        <div className="hidden md:flex space-x-4">
          {routes.map((route) => (
            <Link
              key={route.path}
              to={route.path}
              className={`px-3 py-2 rounded-md text-sm font-medium ${
                location.pathname === route.path
                  ? "bg-blue-600 text-white"
                  : "text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700"
              }`}
            >
              {route.name}
            </Link>
          ))}
                <button
        className="p-2 bg-gray-100 dark:bg-gray-700 rounded-full"
        onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
      >
        {theme === "dark" ? <Sun size={20} /> : <Moon size={20} />}
      </button>
        </div>

        {/* Mobile Menu Button */}
        <button
          className="md:hidden text-gray-800 dark:text-white"
          onClick={() => setMenuOpen(!menuOpen)}
        >
          {menuOpen ? <X /> : <Menu />}
        </button>
      </div>

      {/* Mobile Menu Items */}
      <AnimatePresence>
        {menuOpen && (
          <motion.div
            className="md:hidden bg-white dark:bg-gray-900 border-t border-gray-200 dark:border-gray-800 px-4 pt-2 pb-4 space-y-2"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3 }}
          >
            {routes.map((route) => (
              <Link
                key={route.path}
                to={route.path}
                className={`block px-3 py-2 rounded-md text-sm font-medium ${
                  location.pathname === route.path
                    ? "bg-blue-600 text-white"
                    : "text-gray-800 dark:text-white hover:bg-gray-100 dark:hover:bg-gray-700"
                }`}
                onClick={() => setMenuOpen(false)}
              >
                {route.name}
              </Link>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </nav>
  );
}
