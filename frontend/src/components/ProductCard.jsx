
import { motion } from "framer-motion";
import {
  ShoppingBag,
  Palette,
  Leaf,
  Sparkles,
} from "lucide-react";

export default function ProductCard({ product, index = 0 }) {
  const {
    product_name,
    brand,
    description,
    category,
    color,
    price,
    sustainability_score,
    trend_score,
    semantic_score,
  } = product;

  const formatPrice = (value) =>
    new Intl.NumberFormat("en-IN", {
      style: "currency",
      currency: "INR",
      maximumFractionDigits: 0,
    }).format(value);

  return (
    <motion.article
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: index * 0.07 }}
      className="group overflow-hidden rounded-2xl border border-stone-200 bg-white transition-all duration-300 hover:-translate-y-1 hover:shadow-xl"
    >
      {/* Product visual placeholder */}
      <div className="relative flex aspect-[4/5] items-center justify-center overflow-hidden bg-[#f4f1ec]">
        <div className="absolute left-4 top-4 rounded-full bg-white/90 px-3 py-1.5 text-xs font-medium tracking-wide text-stone-700">
          {category?.replaceAll("_", " ")}
        </div>

        <div className="flex flex-col items-center gap-3 px-6 text-center">
          <ShoppingBag
            size={42}
            strokeWidth={1}
            className="text-stone-400"
          />

          <p className="text-xs uppercase tracking-[0.2em] text-stone-400">
            Product image unavailable
          </p>
        </div>

        {/* Color label */}
        <div className="absolute bottom-4 left-4 right-4">
          <span className="inline-block rounded-full bg-white/90 px-3 py-1.5 text-xs text-stone-600">
            {color || "Color not specified"}
          </span>
        </div>
      </div>

      {/* Product details */}
      <div className="space-y-4 p-5">
        <div className="space-y-1">
          <p className="text-xs uppercase tracking-[0.18em] text-stone-500">
            {brand || "Brand unavailable"}
          </p>

          <h3 className="line-clamp-2 text-base font-medium leading-relaxed text-stone-900">
            {product_name}
          </h3>

          <p className="text-lg font-semibold text-stone-900">
            {price != null ? formatPrice(price) : "Price unavailable"}
          </p>
        </div>

        {description && (
          <p className="line-clamp-3 text-sm leading-6 text-stone-600">
            {description}
          </p>
        )}

        {/* Product attributes */}
        <div className="flex flex-wrap gap-2">
          {product.fit && product.fit !== "unknown" && (
            <span className="rounded-full bg-stone-100 px-3 py-1.5 text-xs text-stone-600">
              {product.fit}
            </span>
          )}
        </div>

        {/* Database details */}
        <div className="grid grid-cols-2 gap-3 border-t border-stone-100 pt-4">
          <div className="space-y-1">
            <p className="flex items-center gap-1 text-xs text-stone-500">
              <Leaf size={13} />
              Sustainability
            </p>

            <p className="text-sm font-medium text-stone-800">
              {sustainability_score ?? "N/A"}
              <span className="text-stone-400"> / 10</span>
            </p>
          </div>

          <div className="space-y-1">
            <p className="flex items-center gap-1 text-xs text-stone-500">
              <Sparkles size={13} />
              Trend score
            </p>

            <p className="text-sm font-medium text-stone-800">
              {trend_score ?? "N/A"}
            </p>
          </div>
        </div>

        {/* Database similarity */}
        {semantic_score != null && (
          <p className="text-xs text-stone-400">
            Database retrieval similarity:{" "}
            {Number(semantic_score).toFixed(4)}
          </p>
        )}

        {/* Dynamic Myntra shopping link */}
        <a
        href={`https://www.myntra.com/search?rawQuery=${encodeURIComponent(
            `${brand || ""} ${product_name || ""}`
            .replace(/\s+/g, " ")
            .trim()
        )}`}
        target="_blank"
        rel="noopener noreferrer"
        className="flex w-full items-center justify-center gap-2 rounded-full bg-[#777b58] px-5 py-3 text-sm font-medium text-white transition hover:bg-[#626648]">
            <ShoppingBag size={16} />
            Find on Myntra
        </a>
      </div>
    </motion.article>
  );
}