package com.stonex.sx3m.analytics.icebergsim;

import com.koloboke.collect.map.ObjObjMap;
import com.stonex.sx3m.dto.SecDefMDUpdate;
import com.stonex.sx3m.dto.bookupdate.BookUpdate;
import com.stonex.sx3m.dto.bookupdate.IcebergUpdate;
import com.stonex.sx3m.dto.bookupdate.IcebergUpdateStatus;
import com.stonex.sx3m.dto.tradeupdate.TradeUpdate;
import com.stonex.sx3m.utils.Nulls;
import com.stonex.sx3m.utils.price.PriceUtils;
import net.openhft.chronicle.wire.LongConversion;
import net.openhft.chronicle.wire.NanoTimestampLongConverter;
import net.openhft.chronicle.wire.SelfDescribingMarshallable;
import org.jetbrains.annotations.Nullable;

import java.util.concurrent.*;

//#todo:
//# - for now, only check icebergs at top of book

/**
 * This is a row in a results table of Iceberg Sims
 */
public class ActiveIceberg extends SelfDescribingMarshallable {

    /**
     * The Iceberg id.
     */
    public long icebergId;
    /**
     * The Is filled.
     */
    public boolean isFilled = false;
    /**
     * The Send order flg.
     */
    public boolean sendOrderFlg = false;
    /**
     * The Send cancel flg.
     */
    public boolean sendCancelFlg = false;
    public boolean isCancelled = false;
    /**
     * The Sim fill exchange time ns.
     */
    @LongConversion(NanoTimestampLongConverter.class)
    public long simFillExchangeTimeNs;

    /**
     * The Latest exchange time ns.
     */
// keep track of last exchangeTimeNs from iceberg, book, and trade updates (for delay + cancel order time calculation)
    @LongConversion(NanoTimestampLongConverter.class)
    public long latestExchangeTimeNs;

    /**
     * The Cancel order exchange time ns.
     */
    @LongConversion(NanoTimestampLongConverter.class)
    public long cancelOrderExchangeTimeNs;
    /**
     * The Send cancel order delay ns.
     */
    public long sendCancelOrderDelayNs = 5_000_000L;

    /**
     * The Md exec.
     */
    public int mdExec = Nulls.INT_NULL;

    /**
     * The Is bid.
     */
    public boolean isBid;
    /**
     * The Price.
     */
    public long price;
    /**
     * The Symbol.
     */
    public String symbol;
    /**
     * The Show size.
     */
    public int showSize;
    /**
     * The Filled size start.
     */
    public int filledSizeStart;
    /**
     * The Last status.
     */
    public IcebergUpdateStatus lastStatus;
    /**
     * The Filled size end.
     */
    public int filledSizeEnd;
    /**
     * The Number reloads.
     */
    public int numberReloads;

    /**
     * The Send order delay ns.
     */
// wait 10 milliseconds (in nanoseconds) before using bookUpdates to establish queue position
    public long sendOrderDelayNs = 5_000_000L;
    /**
     * The Delayed order exchange time ns.
     */
// place order (join queue) after startExchangeTimeNs + delayOrderPlacementNs
    @LongConversion(NanoTimestampLongConverter.class)
    public long delayedOrderExchangeTimeNs;

    /**
     * The Start exchange time ns.
     */
    @LongConversion(NanoTimestampLongConverter.class)
    public long startExchangeTimeNs;

    /**
     * The End exchange time ns.
     */
    @LongConversion(NanoTimestampLongConverter.class)
    public long endExchangeTimeNs;
    /**
     * The Initial book qty.
     */
    public int initialBookQty = Nulls.INT_NULL;

    private ScheduledExecutorService service = Executors.newScheduledThreadPool(4); // calling 4 threads, one is reserved for the cancel action

    /**
     * The Current queue position.
     */
    public int currentQueuePosition = Nulls.INT_NULL;
    /**
     * The Initial queue position.
     */
    public int initialQueuePosition = Nulls.INT_NULL;
    /**
     * The Volume.
     */
    public int volume;

    /**
     * The Tick size.
     */
    public long tickSize;
    /**
     * The Ticks from high.
     */
    public long ticksFromHigh = Nulls.PRICE_NULL;
    /**
     * The Ticks from low.
     */
    public long ticksFromLow = Nulls.PRICE_NULL;
    /**
     * The Is complete.
     */
    public boolean isComplete;
    private boolean priceChanged;
    /**
     * The Iceberg header.
     */
    final static String[] ICEBERG_HEADER = {
            "icebergId", "mdExec", "isBid", "price", "symbol", "showSize",
            "lastStatus", "filledSizeStart", "filledSizeEnd", "numberReloads",
            "startExchangeTimeNs", "delayedOrderExchangeTimeNs", "simFilledExchangeTimeNs", "cancelOrderExchangeTimeNs", "endExchangeTimeNs",
            "initialBookQty", "initialQueuePosition", "currentQueuePosition",
            "volume", "isFilled", "isOrderPlaced", "isCancelled",
            "tickSize", "ticksFromHigh", "ticksFromLow"
    };


    /**
     * Instantiates a new Iceberg simulation result.
     *
     * @param icebergUpdate  the iceberg update
     * @param secDefMDUpdate the sec def md update
     * @param bookUpdate     the book update
     * @param tickSize       the tick size
     */
    public ActiveIceberg(IcebergUpdate icebergUpdate, @Nullable SecDefMDUpdate secDefMDUpdate, @Nullable BookUpdate bookUpdate, long tickSize) {
        this.icebergId = icebergUpdate.icebergId();
        this.isBid = icebergUpdate.bid();
        this.price = icebergUpdate.price();
        this.symbol = icebergUpdate.symbol().toString();
        this.showSize = icebergUpdate.showSize();
        this.filledSizeStart = icebergUpdate.filledQty();
        this.filledSizeEnd = icebergUpdate.filledQty();
        this.numberReloads = 1;
        this.startExchangeTimeNs = icebergUpdate.exchangeTimeNs();
        this.delayedOrderExchangeTimeNs = this.startExchangeTimeNs + sendOrderDelayNs;
        this.endExchangeTimeNs = icebergUpdate.exchangeTimeNs();
        this.lastStatus = icebergUpdate.status();
        this.tickSize = tickSize;

        updateHighLow(secDefMDUpdate);

    }

    /**
     * On trade update.
     *
     * @param tradeUpdate the trade update
     */
    public void onTradeUpdate(TradeUpdate tradeUpdate) {
        this.latestExchangeTimeNs = tradeUpdate.exchangeTimeNs();

        if (tradeUpdate.price() == price)
            volume += tradeUpdate.qty();

//        todo: need?
        if (isFilled)
            return;

        if (tradeUpdate.price() == price && Nulls.isQtyNotNull(currentQueuePosition) && currentQueuePosition != Nulls.INT_NULL) {
            currentQueuePosition -= tradeUpdate.qty();
        }
        if (currentQueuePosition < 0 && currentQueuePosition != Nulls.INT_NULL) {
            isFilled = true;
            simFillExchangeTimeNs = tradeUpdate.exchangeTimeNs();
        }
    }

    /**
     * On book update.
     *
     * @param bookUpdate the book update
     */
    public void onBookUpdate(BookUpdate bookUpdate) {
        this.endExchangeTimeNs = bookUpdate.exchangeTimeNs();
        this.latestExchangeTimeNs = bookUpdate.exchangeTimeNs();

//        wait to send order
        if (!this.sendOrderFlg) {

            if (bookUpdate.exchangeTimeNs() < this.delayedOrderExchangeTimeNs) {
                return;
            } else {
                this.initialBookQty = bookUpdate.getDirectQtyAtPrice(isBid, price, true);
                this.initialQueuePosition = this.initialBookQty + 1;
                this.currentQueuePosition = this.initialQueuePosition;
                this.sendOrderFlg = true;
            }
        }

        if (this.sendOrderFlg) {
// OTHER LOGIC 
        }
    }

    /**
     * On iceberg update.
     *
     * @param icebergUpdate the iceberg update
     */
    public void onIcebergUpdate(IcebergUpdate icebergUpdate) {
        this.endExchangeTimeNs = icebergUpdate.exchangeTimeNs();
        this.latestExchangeTimeNs = icebergUpdate.exchangeTimeNs();

        this.filledSizeEnd = icebergUpdate.filledQty();
        this.lastStatus = icebergUpdate.status();

        if (this.lastStatus == IcebergUpdateStatus.CANCELLED || this.lastStatus == IcebergUpdateStatus.FILLED || this.lastStatus == IcebergUpdateStatus.MISSING_AFTER_RECOVERY) {
            {
                this.checkAndManageOrderState();
            }
            this.isComplete = true;
        }

        if (this.lastStatus == IcebergUpdateStatus.RELOADED || this.lastStatus == IcebergUpdateStatus.LAST_RELOAD) {
            this.numberReloads++;
        }
    }

    /**
     * Send cancel order.
     * <p>
     * //     * @param instrumentSummary the instrument summary
     *
     * @throws Exception the exception
     */
    public void sendCancelOrder() throws Exception {
//        if (this.latestExchangeTimeNs < this.cancelOrderExchangeTimeNs) {
//            return;
//        }
        this.isCancelled = true;
        // todo: timestamp we sent cancel 
//        this.cancelOrderExchangeTimeNs = 
    }

    /**
     * Check and manage order state.
     */
    public void checkAndManageOrderState() {
        if (!this.sendOrderFlg || isComplete || isFilled || isCancelled) {
            return;
        }
//          cancelOrderExchangeTimeNs = latestExchangeTimeNs + sendCancelOrderDelayNs;
        try {
            sendCancelOrder(this.);
        } catch (Exception e) {
            // You might want to do some logging here, or re-throw the exception
            System.out.println("Exception occurred while trying to cancel the order: " + e.getMessage());

        }
    }

    //    todo: WORKING
    public void scheduleCancelOrder() {
        // Calculate the delay time
        long delayMilliseconds = this.sendCancelOrderDelayNs / 1_000_000;

        // Submitting update handling to the service
        service.submit(() -> {
            try {
                // Loop for receiving and handling updates
                if (System.currentTimeMillis() < this.latestExchangeTimeNs + this.sendCancelOrderDelayNs / 1_000_000) {
                    // TODO: receive updates, it depends on your architecture how to get them
                    // Using the implemented methods to handle updates
                    if (tradeUpdate != null) {
                        this.receiveTradeUpdate(tradeUpdate);
                    }
                    if (bookUpdate != null) {
                        this.receiveBookUpdate(bookUpdate);
                    }
                }
            } finally {
                // When the waiting time ends, call sendCancelOrder
                this.sendCancelOrder();
            }
        });
    }

//    /**
//     * Handle updates and schedule cancel.
//     *
//     * @param instrumentSummary the instrument summary
//     * @param bu                the bu
//     * @param tu                the tu
//     * @param iu                the iu
//     */
//    public void handleUpdatesAndScheduleCancel(BookUpdate bu, TradeUpdate tu, IcebergUpdate iu) {
//        CompletableFuture<Void> bookUpdateFuture = CompletableFuture.runAsync(() -> onBookUpdate(bu), service);
//
//        CompletableFuture<Void> tradeUpdateFuture = CompletableFuture.runAsync(() -> onTradeUpdate(tu), service);
//
//        CompletableFuture<Void> icebergUpdateFuture = CompletableFuture.runAsync(() -> onIcebergUpdate(iu), service);
//
//        service.schedule(() -> {
//            try {
//                sendCancelOrder();
//            } catch (Exception e) {
//                e.printStackTrace();
//            } finally {
//                // Shutting down ExecutorService after sendCancelOrder
//                service.shutdown();
//            }
//        }, 5, TimeUnit.MILLISECONDS);
//
//        CompletableFuture.allOf(bookUpdateFuture, tradeUpdateFuture, icebergUpdateFuture); // Do not wait for all futures to complete
//    }


    // update symbol high and low from secDefMdUpdate (if available) or most accurate tracked tradePrice
    private void updateHighLow(@Nullable SecDefMDUpdate secDefMDUpdate) {
        if (secDefMDUpdate == null)
            return;

        final var highPrice = secDefMDUpdate.highTradePrice();
        if (PriceUtils.isNotNull(highPrice))
            ticksFromHigh = (highPrice - price) / tickSize;

        final var lowPrice = secDefMDUpdate.lowTradePrice();
        if (PriceUtils.isNotNull(lowPrice))
            ticksFromLow = (price - lowPrice) / tickSize;
    }
}