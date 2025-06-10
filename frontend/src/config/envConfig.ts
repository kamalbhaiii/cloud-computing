import devEnv from './development.json';
import prodEnv from './production.json';
import type { AppEnv } from '../types/env';

const mode = import.meta.env.MODE || 'development';

const config: AppEnv = mode === 'production' ? prodEnv : devEnv;

export default config;
